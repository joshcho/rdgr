(ns rdgr.cost
  (:require
   [clojure.java.shell :refer [sh]]
   [clojure.string :as str]
   [rdgr.utils :as u]
   [clojure.spec.alpha :as s]))

(defn get-num-tokens [model text]
  (let [{:keys [out err exit]} (sh "python" "token_counter.py" text model)]
    (if (zero? exit)
      (Integer/parseInt (str/trim out))
      (throw (Exception. err)))))

(s/def :model/spec
  #(u/in? ["gpt-4" "gpt-3.5-turbo"] %))
(s/def ::cost float?)
(s/def :cost/prompt ::cost)
(s/def :cost/completion ::cost)
(s/def :cost/type #(u/in? [:cost/prompt :cost/completion] %))
(s/def :cost/spec
  (s/keys :req [:cost/prompt
                :cost/completion]))
;; https://openai.com/pricing
(def pricing {"gpt-4"
              {:cost/prompt 0.03
               :cost/completion 0.06}
              "gpt-3.5-turbo"
              {:cost/prompt 0.002
               :cost/completion 0.002}})
(s/assert (s/map-of :model/spec
                    :cost/spec)
          pricing)

(defn get-cost [problem]
  {:pre [(s/valid? :problem/completed-spec problem)]}
  (let [{:problem/keys [model question raw-completion]} problem]
    (+
     (*
      (/ (get-in pricing [model :cost/prompt]) 1000.0)
      (get-num-tokens model question))
     (*
      (/ (get-in pricing [model :cost/completion]) 1000.0)
      (get-num-tokens model raw-completion)))))