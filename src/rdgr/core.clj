(ns rdgr.core
  (:require
   [wkok.openai-clojure.api :as api]
   [clojure.string :as str]
   [rdgr.data :as data]
   [rdgr.cost :as cost]
   ;; [clojure.walk :as walk]
   [clojure.spec.alpha :as s]))
(s/check-asserts true)
(api/create-chat-completion
 {:model "gpt-3.5-turbo"
  :messages [{:role "system" :content "You are a helpful assistant."}
             {:role "user" :content "Who won the world series in 2020?"}
             {:role "assistant" :content "The Los Angeles Dodgers won the World Series in 2020."}
             {:role "user" :content "Where was it played?"}]})

(def data
  [{:data/name "John" :age 30 :city "New York"}
   {:data/name "Jane" :age 25 :city "Los Angeles"}
   {:data/name "Bob" :age 35 :city "Chicago"}
   ])
(data/read-csv "file")
(data/overwrite-csv "file" data)
(data/append-csv "file" data)
(cost/get-cost "gpt-4"
               (apply str
                      (repeat 1000
                              "hello there! i am curious"))
               :cost/completion)

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
