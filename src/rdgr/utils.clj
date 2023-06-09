(ns rdgr.utils
  (:require
   [clojure.java.shell :refer [sh]]
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
         (string? input)]}
  (let [temp-file (io/file "temp.py")
        _ (spit temp-file python-code)
        executor (Executors/newSingleThreadExecutor)
        future (-> executor
                   (.submit
                    (reify Callable
                      (call [_] (sh "python" (.getAbsolutePath temp-file) :in input)))))]
    (try
      (.get future TIMEOUT TimeUnit/SECONDS)
      (catch java.util.concurrent.TimeoutException e
        (.shutdownNow executor)
        nil))))
