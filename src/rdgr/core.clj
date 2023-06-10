(ns rdgr.core
  (:require
   [clojure.spec.alpha :as s]
   [clojure.java.shell :refer [sh]]
   [clojure.string :as str]
   [jsonista.core :as json]
   [clojure.java.io :as io]
   [rdgr.cost :as cost]
   [rdgr.utils :as u]
   [rdgr.data-csv :as data-csv]
   [rdgr.data :as data]
   [wkok.openai-clojure.api :as api]))
(import 'java.util.concurrent.Executors)

(s/check-asserts true)
(def problems-conn (atom {"gpt-3.5-turbo" {}
                          "gpt-4" {}}))
;; (reset! problems-conn (read-string (slurp "./gen/saved.txt")))

(defn how-prompt-fn [CoT-instruction format-prompt]
  (if CoT-instruction
    (format
     "Simplify your answer as much as possible. %s Box your answer, i.e. \\boxed{answer}.%s" CoT-instruction
     (or format-prompt ""))
    (format
     "Simplify your answer as much as possible. Box your answer, i.e. \\boxed{answer}.%s"
     format-prompt)))

(defn solve-prompt-format-fn [CoT-instruction]
  (format "Given a mathematics problem, determine the answer. %s"
          (how-prompt-fn CoT-instruction nil)))

(def base-prompt
  (solve-prompt-format-fn "Do not show your work."))

(def CoT-prompt
  (solve-prompt-format-fn "Think step by step."))
(defn CoT-alternative-prompt-fn [problem]
  (format
   "Solve the problem carefully. Put the final answer in \\boxed{}. %s"
   problem))

(def show-your-work-prompt
  (solve-prompt-format-fn "Show your work."))

(def plan-prompt
  "Given a mathematics problem, figure out the plan for solving this problem, but do not solve it.")

(def solve-with-plan-prompt
  (format "Given a mathematics problem and a plan on how to solve the problem, determine the answer by following the plan. %s"
          (how-prompt-fn nil nil)))

(def solve-with-counterfactual-prompt
  (how-prompt-fn "Think step by step. Each step should contain Step N, Counterfactual N, and Revised Step N. In the counterfactual, assume the step is incorrect, and extrapolate why."
                 "\n\nFormat:
Step 1: {step 1}
Counterfactual 1: If the above step is incorrect, it's because {counterfactual 1}.
Revised Step 1: {revised step 1}
..."))

(def approach-prompt
  "Given a mathematics problem, figure out many possible approaches for solving this problem, but do not solve it. Afterwards, suggest the best approach.")

(defn get-completion-text [model content]
  (s/assert :model/spec model)
  (assert (string? content))
  (get-in
   (api/create-chat-completion
    {:model model
     :messages [{:role "system" :content "You are CodeAI. Respond with two parts: some reasoning and a single Python code block starting with ```python. End after ```."}
                {:role "user" :content content}]})
   [:choices 0 :message :content]))

(defn populate-problem [model id]
  (s/assert :model/spec model)
  (assert (<= 0 id 4999))
  (let [{:problem/keys [question] :as problem}
        (data/get-problem data/default-split id)]
    (s/assert :problem/spec problem)
    (if (get-in @problems-conn [model id])
      (do (u/safe-print (format "problem for %s with id %s already populated" model id))
          nil)
      (do
        (u/safe-print (format "starting completion for %s with id %s" model id))
        (let [raw-completion-text (get-completion-text model question)]
          (u/safe-print (format "finished completion for %s with id %s" model id))
          (swap!
           problems-conn
           (fn [problems-db]
             (assoc-in
              problems-db
              [model id]
              (into
               problem
               {:problem/model model
                :problem/raw-completion raw-completion-text
                :problem/completion (-> raw-completion-text
                                        (str/replace-first #"\p{all}*```python *" "")
                                        (str/replace-first #"```\p{all}*" ""))})))))
        true))))

(defn repopulate-problem [model id]
  (s/assert :model/spec model)
  (assert (<= 0 id 4999))
  (swap! problems-conn
         (fn [problems-db]
           (update problems-db model
                   (fn [model-map]
                     (dissoc model-map id)))))
  (populate-problem model id)
  true)

;; (for [i (range 1 10)]
;;   (when (get-in @problems-conn "gpt-3.5-turbo" i)
;;     (swap! problems-conn
;;            (fn [problems-db]
;;              (update-in
;;               problems-db
;;               ["gpt-3.5-turbo" i]
;;               #(clojure.set/rename-keys % {:difficulty :problem/difficulty
;;                                        :url :url/difficulty}))))))

(defn parallel-do-problem [model ids func]
  {:pre [(s/valid? :model/spec model)
         (s/valid? (s/coll-of :problem/id) ids)]}
  (let [executor (Executors/newFixedThreadPool 10) ; adjust thread pool size to your needs
        tasks (map (fn [id]
                     (.submit executor
                              (reify java.util.concurrent.Callable
                                (call [_] (func model id)))))
                   ids)]
    ;; now we wait for all tasks to complete and collect the results
    (doall (map (fn [future] (.get future)) tasks))
    true))

(defn parallel-populate-problem [model ids]
  {:pre [(s/valid? :model/spec model)
         (s/valid? (s/coll-of :problem/id) ids)]}
  (parallel-do-problem model ids populate-problem))

(defn evaluate-problem [model id]
  {:pre [(s/valid? :model/spec model)
         (s/valid? :problem/id id)]}
  (if-let [problem (get-in @problems-conn [model id])]
    (if (:problem/evaluations problem)
      (do (u/safe-print (format "problem for %s with id %s already has eval" model id))
          nil)
      (let [{:problem/keys [completion input-output]} problem
            {:keys [inputs outputs]} input-output]
        (assert (and completion inputs outputs))
        (assert (= (count inputs) (count outputs)))
        (u/safe-print (format "starting evaluation for %s with id %s" model id))
        (let [result
              (doall
               (->> (map vector inputs outputs)
                    (pmap (fn [[input output]]
                            (some->>
                             ;; some->> since execute-python-code returns nil on timeout
                             (u/execute-python-code completion input)
                             :out
                             (= output))))))]
          (u/safe-print (format "finished evaluation for %s with id %s" model id))
          (swap! problems-conn
                 (fn [problems-db]
                   (assoc-in
                    problems-db [model id :problem/evaluations]
                    result))))
        true))
    (do (u/safe-print "problem not found")
        nil)))

(defn get-completion [model id]
  (get-in @problems-conn [model id :problem/completion]))
(defn get-question [model id]
  (get-in @problems-conn [model id :problem/question]))
(defn get-raw-completion [model id]
  (get-in @problems-conn [model id :problem/raw-completion]))
(defn get-evaluations [model id]
  (get-in @problems-conn [model id :problem/evaluations]))
(defn get-difficulty [model id]
  (get-in @problems-conn [model id :problem/difficulty]))

(defn get-accuracy [model id]
  {:pre [(s/valid? :model/spec model)
         (s/valid? :problem/id id)]}
  (when-let [evaluations (get-in @problems-conn [model id :problem/evaluations])]
    (/ (count (filter identity evaluations))
       (count evaluations))))
(defn get-accuracies [model ids]
  (map (fn [id]
         (get-accuracy model id))
       ids))

(defn get-average-accuracy [model ids]
  (assert
   (every? :problem/evaluations
           (map (fn [id] (get-in @problems-conn [model id]))
                ids)))
  (float
   (/ (apply + (map (fn [id]
                      (get-accuracy model id))
                    ids))
      (count ids))))
(defn get-accuracy-difference [model1 model2 ids]
  (map vector
       ids
       (map -
            (map (fn [id] (get-accuracy model1 id))
                 ids)
            (map (fn [id] (get-accuracy model2 id))
                 ids))))

(defn parallel-evaluate-problem [model ids]
  {:pre [(s/valid? :model/spec model)
         (s/valid? (s/coll-of :problem/id) ids)]}
  (parallel-do-problem model ids evaluate-problem))

(defn get-problem-cost [model id]
  (if-let [{old-cost :problem/cost :as problem} (get-in @problems-conn [model id])]
    (if old-cost
      (do (u/safe-print "problem cost already exists")
          old-cost)
      (let [computed-cost (cost/get-cost problem)]
        (swap! problems-conn
               (fn [problems-db]
                 (assoc-in
                  problems-db [model id :problem/cost]
                  computed-cost)))))
    (do (u/safe-print "problem not found")
        nil)))

(defn get-problem-total-cost [model ids]
  {:pre [(s/valid? :model/spec model)
         (s/valid? (s/coll-of :problem/id) ids)]}
  (assert
   (every? identity
           (map (fn [id] (get-in @problems-conn [model id]))
                ids)))
  (apply + (pmap (fn [id]
                   (get-problem-cost model id))
                 ids)))

;; (spit "./gen/saved.txt" (prn-str @problems-conn))
;; (def my-map (read-string (slurp "./gen/saved.txt")))

;; (data-csv/read-csv "file")
;; (data-csv/overwrite-csv "gpt-4-saved" (vals (get @problems-conn "gpt-4")))
;; (data-csv/append-csv "file" data)
;; (cost/get-cost "gpt-4"
;;                (apply str
;;                       (repeat 1000
;;                               "hello there! i am curious"))
;;                :cost/completion)

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
