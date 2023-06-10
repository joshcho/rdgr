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
;; (def problems-conn (atom {"gpt-3.5-turbo" {}
;;                           "gpt-4" {}}))

;; (reset! problems-conn (read-string (slurp "./gen/saved.txt")))

(def dataset-conn (atom nil))
(reset! dataset-conn
        (mapv
         (fn [problem-map]
           (assoc problem-map
                  :answer
                  (u/find-boxed (:solution problem-map))))
         ;; (data-csv/read-csv "../data/MATH-dispatch/dispatch-test-1000.csv")
         (data-csv/read-csv "dispatch-test-1000")))
(def dataset-saved-conn dataset-conn)

;; (data-csv/overwrite-csv "dispatch-test-1000" @dataset-conn)

(defn get-completion-text [completion-model prompt]
  {:pre [(u/in? '("gpt-4" "gpt-3.5-turbo") completion-model)]}
  (assert (string? prompt))
  (let [res (get-in
             (api/create-chat-completion
              {:model completion-model
               :messages [{:role "user"
                           :content prompt}]
               :max_tokens 2000})
             [:choices 0 :message :content])]
    (when (nil? res)
      (u/safe-print (format "nil returned for %s" prompt)))
    res))

(defn parallel-do-problem [model start-idx end-idx func]
  {:pre [(s/valid? :model/spec model)
         (int? start-idx)
         (int? end-idx)]}
  (let [executor (Executors/newFixedThreadPool 25) ; adjust thread pool size to your needs
        tasks (doall (map (fn [idx]
                            (.submit executor (cast Callable
                                                    (fn [] (func model idx)))))
                          (range start-idx end-idx)))]
    ;; now we wait for all tasks to complete and collect the results
    (future
      (doall (map (fn [future] (.get future)) tasks))
      (.shutdown executor)
      (u/safe-print (format "** finished completions for %s from %d to %d **"
                            model start-idx end-idx)))
    (future
      (Thread/sleep 120000)
      (.shutdownNow executor)
      ;; forcibly shut down
      )
    ;; shut down the executor service
    true))
;; (->> (range 10)
;;      (map
;;       #(get @dataset-conn %))
;;      (map
;;       (fn [problem-map]
;;         [(cost/get-num-tokens "gpt-4" (:gpt-4 problem-map))
;;          (cost/get-num-tokens "gpt-3.5-turbo" (:gpt-3.5-turbo problem-map))])))

(defn complete-problem
  ([model start-idx end-idx]
   {:pre [(s/valid? :model/spec model)
          (int? start-idx)
          (int? end-idx)]}
   (parallel-do-problem model start-idx end-idx complete-problem))
  ([model idx]
   {:pre [(s/valid? :model/spec model)
          (int? idx)]}
   (let [{:keys [problem] :as problem-map}
         (get @dataset-conn idx)
         comp-key (keyword model)]
     (if (get problem-map comp-key)
       (do (u/safe-print (format "already has %s completion for %d" model idx))
           nil)
       (do
         (u/safe-print (format "starting %s completion for %d" model idx))
         (let [prompt (u/construct-prompt model problem)
               received-completion (get-completion-text
                                    (u/get-completion-model model)
                                    prompt)
               answer-key (keyword (format "%s-answer" model))
               cost-key (keyword (format "%s-cost" model))]
           (if (= received-completion :timeout)
             (do
               (u/safe-print (format "timed out on %s completion for %d" model idx))
               nil)
             (let [answer (u/find-boxed received-completion)
                   ;; cost (cost/get-cost model prompt received-completion)
                   ]
               ;; (when (or (int? cost)
               ;;           (or (string? answer) (nil? answer))
               ;;           (not (string? received-completion)))
               ;;   (u/safe-print (format "sth went wrong %d \n%s\n%s"
               ;;                         cost answer received-completion)))
               (swap!
                dataset-conn
                (fn [dataset-db]
                  (assoc
                   dataset-db idx
                   (into
                    problem-map
                    {comp-key received-completion
                     answer-key answer
                     ;; cost-key cost
                     }))))
               (u/safe-print (format "finished %s completion for %d" model idx)))))
         true)))))

;; (defn complete-dataset [model]
;;   {:pre [(s/valid? :model/spec model)]}

;;   (cond (or (= model "gpt-3.5-turbo")
;;             (= model "gpt-4"))

;;         :else
;;         (assert false))
;;   (let [topic-dir (format "./data/MATH/%s/%s" split-name topic)
;;         ;; metadata (read-json (format "%s/metadata.json" dir-name))
;;         ;; input-output (read-json (format "%s/input_output.json" dir-name))
;;         ;; question (read-txt (format "%s/question.txt" dir-name))
;;         ;; solutions (read-json (format "%s/solutions.json" dir-name))
;;         ]
;;     (->>
;;      (map #(.getName %)
;;           (.listFiles (File. topic-dir)))
;;      (map
;;       (fn [filename]
;;         (->
;;          (format "%s/%s" topic-dir filename)
;;          read-json)))
;;      (take 10)
;;      (data-csv/overwrite-csv "testing-10"))
;;     ;; (->
;;     ;;  (rename-keys metadata
;;     ;;               {:difficulty :problem/difficulty
;;     ;;                :url :problem/url})
;;     ;;  (assoc
;;     ;;   :problem/id id
;;     ;;   :problem/question question
;;     ;;   :problem/solutions solutions
;;     ;;   :problem/input-output input-output))
;;     ))

;; (defn repopulate-problem [model id]
;;   (s/assert :model/spec model)
;;   (assert (<= 0 id 4999))
;;   (swap! problems-conn
;;          (fn [problems-db]
;;            (update problems-db model
;;                    (fn [model-map]
;;                      (dissoc model-map id)))))
;;   (populate-problem model id)
;;   true)

;; (for [i (range 1 10)]
;;   (when (get-in @problems-conn "gpt-3.5-turbo" i)
;;     (swap! problems-conn
;;            (fn [problems-db]
;;              (update-in
;;               problems-db
;;               ["gpt-3.5-turbo" i]
;;               #(clojure.set/rename-keys % {:difficulty :problem/difficulty
;;                                        :url :url/difficulty}))))))



;; (defn parallel-populate-problem [model ids]
;;   {:pre [(s/valid? :model/spec model)
;;          (s/valid? (s/coll-of :problem/id) ids)]}
;;   (parallel-do-problem model ids populate-problem))

;; (defn evaluate-problem [model id]
;;   {:pre [(s/valid? :model/spec model)
;;          (s/valid? :problem/id id)]}
;;   (if-let [problem (get-in @problems-conn [model id])]
;;     (if (:problem/evaluations problem)
;;       (do (u/safe-print (format "problem for %s with id %s already has eval" model id))
;;           nil)
;;       (let [{:problem/keys [completion input-output]} problem
;;             {:keys [inputs outputs]} input-output]
;;         (assert (and completion inputs outputs))
;;         (assert (= (count inputs) (count outputs)))
;;         (u/safe-print (format "starting evaluation for %s with id %s" model id))
;;         (let [result
;;               (doall
;;                (->> (map vector inputs outputs)
;;                     (pmap (fn [[input output]]
;;                             (some->>
;;                              ;; some->> since execute-python-code returns nil on timeout
;;                              (u/execute-python-code completion input)
;;                              :out
;;                              (= output))))))]
;;           (u/safe-print (format "finished evaluation for %s with id %s" model id))
;;           (swap! problems-conn
;;                  (fn [problems-db]
;;                    (assoc-in
;;                     problems-db [model id :problem/evaluations]
;;                     result))))
;;         true))
;;     (do (u/safe-print "problem not found")
;;         nil)))

;; (defn get-completion [model id]
;;   (get-in @problems-conn [model id :problem/completion]))
;; (defn get-question [model id]
;;   (get-in @problems-conn [model id :problem/question]))
;; (defn get-raw-completion [model id]
;;   (get-in @problems-conn [model id :problem/raw-completion]))
;; (defn get-evaluations [model id]
;;   (get-in @problems-conn [model id :problem/evaluations]))
;; (defn get-difficulty [model id]
;;   (get-in @problems-conn [model id :problem/difficulty]))

;; (defn get-accuracy [model id]
;;   {:pre [(s/valid? :model/spec model)
;;          (s/valid? :problem/id id)]}
;;   (when-let [evaluations (get-in @problems-conn [model id :problem/evaluations])]
;;     (/ (count (filter identity evaluations))
;;        (count evaluations))))
;; (defn get-accuracies [model ids]
;;   (map (fn [id]
;;          (get-accuracy model id))
;;        ids))

;; (defn get-average-accuracy [model ids]
;;   (assert
;;    (every? :problem/evaluations
;;            (map (fn [id] (get-in @problems-conn [model id]))
;;                 ids)))
;;   (float
;;    (/ (apply + (map (fn [id]
;;                       (get-accuracy model id))
;;                     ids))
;;       (count ids))))
;; (defn get-accuracy-difference [model1 model2 ids]
;;   (map vector
;;        ids
;;        (map -
;;             (map (fn [id] (get-accuracy model1 id))
;;                  ids)
;;             (map (fn [id] (get-accuracy model2 id))
;;                  ids))))

;; (defn parallel-evaluate-problem [model ids]
;;   {:pre [(s/valid? :model/spec model)
;;          (s/valid? (s/coll-of :problem/id) ids)]}
;;   (parallel-do-problem model ids evaluate-problem))

;; (defn get-problem-cost [model id]
;;   (if-let [{old-cost :problem/cost :as problem} (get-in @problems-conn [model id])]
;;     (if old-cost
;;       (do (u/safe-print "problem cost already exists")
;;           old-cost)
;;       (let [computed-cost (cost/get-cost problem)]
;;         (swap! problems-conn
;;                (fn [problems-db]
;;                  (assoc-in
;;                   problems-db [model id :problem/cost]
;;                   computed-cost)))))
;;     (do (u/safe-print "problem not found")
;;         nil)))

;; (defn get-problem-total-cost [model ids]
;;   {:pre [(s/valid? :model/spec model)
;;          (s/valid? (s/coll-of :problem/id) ids)]}
;;   (assert
;;    (every? identity
;;            (map (fn [id] (get-in @problems-conn [model id]))
;;                 ids)))
;;   (apply + (pmap (fn [id]
;;                    (get-problem-cost model id))
;;                  ids)))

;; ;; (spit "./gen/saved.txt" (prn-str @problems-conn))
;; ;; (def my-map (read-string (slurp "./gen/saved.txt")))

;; ;; (data-csv/read-csv "file")
;; ;; (data-csv/overwrite-csv "gpt-4-saved" (vals (get @problems-conn "gpt-4")))
;; ;; (data-csv/append-csv "file" data)
;; ;; (cost/get-cost "gpt-4"
;; ;;                (apply str
;; ;;                       (repeat 1000
;; ;;                               "hello there! i am curious"))
;; ;;                :cost/completion)

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
