(ns rdgr.utils
  (:require
   [clojure.spec.alpha :as s]
   [clojure.java.shell :refer [sh]]
   [clojure.string :as str]
   [clojure.java.io :as io])
  (:import [java.util.concurrent Executors TimeUnit Callable]))

(def print-lock (Object.))

(defn safe-print [& args]
  (locking print-lock
    (apply println args)))

(defn in?
  "true if coll contains elm"
  [coll elm]
  {:pre [(coll? coll)]}
  (some #(= elm %) coll))

(def TIMEOUT 10)

(defn execute-python-code [python-code input]
  {:pre [(string? python-code)
         (or (string? input)
             (nil? input))]}
  (let [temp-file (io/file "temp.py")
        _ (spit temp-file python-code)
        executor (Executors/newSingleThreadExecutor)
        future (-> executor
                   (.submit
                    (reify Callable
                      (call [_]
                        (if input
                          (sh "python" (.getAbsolutePath temp-file) :in input)
                          (sh "python" (.getAbsolutePath temp-file)))))))]
    (try
      (.get future TIMEOUT TimeUnit/SECONDS)
      (catch java.util.concurrent.TimeoutException e
        (.shutdownNow executor)
        nil))))

(defn find-boxed [s]
  (let [start-index (+ 7 (.indexOf s "\\boxed{"))]
    (if (neg? start-index)
      nil
      (if (neg? (.indexOf (subs s start-index) "\\boxed{"))
        (loop [counter 1
               end-index start-index]
          (let [current-char (get s end-index)]
            (cond
              (= current-char \{) (recur (inc counter) (inc end-index))
              (= current-char \})
              (if (= (dec counter) 0)
                (subs s start-index end-index)
                (recur (dec counter) (inc end-index)))
              (nil? current-char) nil
              :else
              (recur counter (inc end-index)))))
        (find-boxed (subs s start-index))))))

(defn construct-prompt [model problem]
  {:pre [(s/valid? :model/spec model)
         (string? problem)]}
  (cond
    (in? '("gpt-4" "gpt-3.5-turbo") model)
    (format "Put the final answer in \\boxed{}. %s" problem)
    (= model "gpt-4-CoT")
    (format
     "Solve the problem carefully. Put the final answer in \\boxed{}. %s"
     problem)
    (= model "gpt-4-PS")
    (format
     "import math
import numpy as np
import sympy as sp
# Question: %s
# Answer this question by implementing a solver() function.
def solver():
    # Let's write a Python program step by step, and then return the answer
    # Firstly, we need to define the following variable(s):\n"
     problem)
    :else
    (assert false)))

(defn answer-equiv? [answer1 answer2]
  {:pre [(string? answer1)
         (string? answer2)]}
  (let [{:keys [out err exit]} (sh "python" "math_equivalence.py"
                                   (format "\"%s\"" answer1)
                                   (format "\"%s\"" answer2))]
    (if (zero? exit)
      (Boolean/parseBoolean (str/trim out))
      (throw (Exception. err)))))

(defn get-completion-model [model]
  {:pre [(s/valid? :model/spec model)]}
  (case model
    "gpt-4" "gpt-4"
    "gpt-3.5-turbo" "gpt-3.5-turbo"
    "gpt-4-CoT" "gpt-4"
    "gpt-4-PS" "gpt-4"
    (assert false)))

(defn get-answer [model completion]
  (cond (in? '("gpt-3.5-turbo"
               "gpt-4"
               "gpt-4-CoT")
             model)
        (find-boxed completion)
        (= model "gpt-4-PS")
        (:out
         (execute-python-code
          (format "import math
import numpy as np
import sympy as sp
def solver():\n%s\nprint(solver(), end=\"\")" completion)
          nil))
        :else
        (assert false)))

(defn get-model-key [model]
  {:pre [(s/valid? :model/spec model)]}
  (case model
    "gpt-3.5-turbo" :gpt-3.5-turbo
    "gpt-4" :gpt-4
    "gpt-4-CoT" :gpt-4-CoT
    "gpt-4-PS" :gpt-4-PS
    (assert false)))

(defn get-model-answer-key [model]
  {:pre [(s/valid? :model/spec model)]}
  (case model
    "gpt-3.5-turbo" :gpt-3.5-turbo-answer
    "gpt-4" :gpt-4-answer
    "gpt-4-CoT" :gpt-4-CoT-answer
    "gpt-4-PS" :gpt-4-PS-answer
    (assert false)))

(defn check-missing [model start-idx end-idx]
  {:pre [(s/valid? :model/spec model)]}
  (->> (range start-idx end-idx)
       (map #(vector % (get @rdgr.core/dataset-conn %)))
       (remove (fn [[idx problem-map]]
                 (get problem-map
                      (get-model-key model))))
       (map first)))
(defn check-diff-missing [start-idx end-idx]
  ;; {:pre [(s/valid? :model/spec model)]}
  (->> (range start-idx end-idx)
       (map #(vector % (get @rdgr.core/dataset-conn %)))
       (remove (fn [[idx problem-map]]
                 (get problem-map :pred-difficulty)))
       (map first)))

(defn get-incorrect-ids [model start-idx end-idx]
  {:pre [(s/valid? :model/spec model)]}
  (->> (range start-idx end-idx)
       (map #(vector % (get @rdgr.core/dataset-conn %)))
       (remove (fn [[idx problem-map]]
                 (let [model-answer (get problem-map
                                         (get-model-answer-key model))
                       answer (get problem-map :answer)]
                   (when (and model-answer answer)
                     (answer-equiv? model-answer answer)))))
       (map first)))
(count )
(def incorrect-ids-gpt-4 (get-incorrect-ids "gpt-4" 0 500))
(def incorrect-ids-gpt-4-CoT (get-incorrect-ids "gpt-4-CoT" 0 500))
(def incorrect-ids-gpt-3-5-turbo (get-incorrect-ids "gpt-3.5-turbo" 0 500))
(def intersect-incorrect-ids
  (clojure.set/intersection
   (set incorrect-ids-gpt-3-5-turbo)
   (set incorrect-ids-gpt-4-CoT)
   (set incorrect-ids-gpt-4)))

(defn average [xs]
  (/ (apply + xs)
     (count xs)))

(def difficulty-system-prompt "Determine the difficulty of the high school problem. The difficulty is either AMC10, AMC12, or AIME.

Respond in the following format:
Justification: {justification}
Difficulty: {difficulty}")

;; (->> intersect-incorrect-ids
;;      (map
;;       #(get @rdgr.core/dataset-conn %))
;;      (map :level)
;;      (map #(Float/parseFloat %))
;;      average)
;; (->> incorrect-ids-gpt-3-5-turbo
;;      (map
;;       #(get @rdgr.core/dataset-conn %))
;;      (map :level)
;;      (map #(Float/parseFloat %))
;;      average)
;; (->> (range 200)
;;      (map
;;       #(get @rdgr.core/dataset-conn %))
;;      (map :level)
;;      (map #(Float/parseFloat %))
;;      average)

;; (defn get-levels [start-idx end-idx]
;;   {:pre [(int? start-idx)
;;          (int? end-idx)]}
;;   (->> (range start-idx end-idx)
;;        (map #(:level (get @rdgr.core/dataset-conn %)))))

;; (defn incorrect?)

;; (defn get-info
;;   ([start-idx end-idx]
;;    (get-info start-idx end-idx nil nil))
;;   ([start-idx end-idx subkeys]
;;    (get-info start-idx end-idx subkeys nil))
;;   ([start-idx end-idx subkeys filter-fn]
;;    (let [filter-fn (or filter-fn
;;                        (fn [x]
;;                          true))]
;;      (->> (range start-idx end-idx)
;;           (map (fn [idx]
;;                  (into
;;                   {:idx idx}
;;                   (get @rdgr.core/dataset-conn idx))))
;;           (filter filter-fn)
;;           (map (fn [problem-map]
;;                  (if subkeys
;;                    (select-keys problem-map subkeys)
;;                    problem-map)))))))

;; (defn how-prompt-fn [CoT-instruction format-prompt]
;;   (if CoT-instruction
;;     (format
;;      "Simplify your answer as much as possible. %s Box your answer, i.e. \\boxed{answer}.%s" CoT-instruction
;;      (or format-prompt ""))
;;     (format
;;      "Simplify your answer as much as possible. Box your answer, i.e. \\boxed{answer}.%s"
;;      format-prompt)))

;; (defn solve-prompt-format-fn [CoT-instruction]
;;   (format "Given a mathematics problem, determine the answer. %s"
;;           (how-prompt-fn CoT-instruction nil)))

;; (def base-prompt
;;   (solve-prompt-format-fn "Do not show your work."))

;; (def CoT-prompt
;;   (solve-prompt-format-fn "Think step by step."))
;; (defn CoT-alternative-prompt-fn [problem]
;;   (format
;;    "Solve the problem carefully. Put the final answer in \\boxed{}. %s"
;;    problem))

;; (def show-your-work-prompt
;;   (solve-prompt-format-fn "Show your work."))

;; (def plan-prompt
;;   "Given a mathematics problem, figure out the plan for solving this problem, but do not solve it.")

;; (def solve-with-plan-prompt
;;   (format "Given a mathematics problem and a plan on how to solve the problem, determine the answer by following the plan. %s"
;;           (how-prompt-fn nil nil)))

;; (def solve-with-counterfactual-prompt
;;   (how-prompt-fn "Think step by step. Each step should contain Step N, Counterfactual N, and Revised Step N. In the counterfactual, assume the step is incorrect, and extrapolate why."
;;                  "\n\nFormat:
;; Step 1: {step 1}
;; Counterfactual 1: If the above step is incorrect, it's because {counterfactual 1}.
;; Revised Step 1: {revised step 1}
;; ..."))

;; (def approach-prompt
;;   "Given a mathematics problem, figure out many possible approaches for solving this problem, but do not solve it. Afterwards, suggest the best approach.")
