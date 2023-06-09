(ns rdgr.data
  (:require
   [jsonista.core :as json]
   [clojure.string :as str]
   [clojure.java.io :as io]
   [clojure.walk :as walk]
   [rdgr.utils :as u]
   [clojure.spec.alpha :as s]
   [clojure.set :refer (rename-keys)]
   [medley.core :refer (map-vals)]))

(def default-split "test")
(defn read-json [filename]
  (let [file (io/file filename)]
    (->> file
         json/read-value
         walk/keywordize-keys)))

(defn get-difficulty-ids []
  (->> (range 0 4999)
       (map (fn [id]
              (let [difficulty
                    (->> id
                         (format "./data/APPS/%s/%04d/metadata.json"
                                 default-split)
                         read-json
                         :difficulty)]
                [difficulty id])))
       (group-by first)
       (map-vals (fn [xs]
                   (map second xs)))))

(def difficulty-ids (get-difficulty-ids))

(defn read-txt [filename]
  (str/trim (slurp filename)))

(s/def :split/spec #(u/in? '("train" "test") %))
(s/def :difficulty/spec #(u/in? '("introductory" "interview" "competition") %))
(s/def :problem/id #(<= 0 % 4999))
(s/def :problem/question string?)
(s/def :problem/model :model/spec)
(s/def :problem/raw-completion string?)
(s/def :problem/completion string?)
(s/def :problem/evaluations (s/coll-of #(or (nil? %)
                                            (boolean? %))))
(s/def ::inputs (s/coll-of string?))
(s/def ::outputs (s/coll-of string?))
(s/def :problem/input-output (s/keys :req-un [::inputs
                                              ::outputs]))
(s/def :problem/spec (s/keys :req [:problem/id
                                   :problem/question
                                   :problem/input-output]
                             :opt [:problem/model
                                   :problem/raw-completion
                                   :problem/completion
                                   :problem/evaluations]))
(s/def :problem/completed-spec (s/keys :req [:problem/id
                                             :problem/question
                                             :problem/input-output
                                             :problem/model
                                             :problem/raw-completion
                                             :problem/completion]
                                       :opt [:problem/evaluations]))

(defn get-problem [split-name id]
  (let [dir-name (format "./data/APPS/%s/%04d" split-name id)
        metadata (read-json (format "%s/metadata.json" dir-name))
        input-output (read-json (format "%s/input_output.json" dir-name))
        question (read-txt (format "%s/question.txt" dir-name))
        solutions (read-json (format "%s/solutions.json" dir-name))
        ]
    (->
     (rename-keys metadata
                  {:difficulty :problem/difficulty
                   :url :problem/url})
     (assoc
      :problem/id id
      :problem/question question
      :problem/solutions solutions
      :problem/input-output input-output))))

(defn get-problem-of-difficulty [split-name difficulty n]
  {:pre [(s/valid? :split/spec split-name)
         (s/valid? :difficulty/spec difficulty)]}
  ;; for now
  (assert (= split-name "test"))
  ;; n is not id, it's the nth in the difficulty
  (when-let [id (nth (get difficulty-ids difficulty)
                     n)]
    (let [problem (get-problem split-name id)]
      (assert (= (:problem/difficulty problem)
                 difficulty))
      problem)))
