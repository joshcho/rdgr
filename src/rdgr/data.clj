(ns rdgr.data
  (:require
   [clojure.string :as str]
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.walk :as walk]
   [clj-time.format]
   [clj-time.core]
   [clojure.spec.alpha :as s]))

(defn keyword-to-string [kw]
  (if (namespace kw)
    (str (namespace kw) "/" (name kw))
    (name kw)))
(def data-dir "./gen/")
(def temp-dir "./gen/temp/")
(def datetime-format
  (clj-time.format/with-zone
    (clj-time.format/formatter "MMdd_HHmmss")
    (clj-time.core/default-time-zone)))

(defn process-filename [raw-filename]
  (str data-dir
       (if (str/ends-with? raw-filename ".csv")
         raw-filename
         (format "%s.csv" raw-filename))))
(defn process-temp-filename [raw-filename]
  (let [timestamp (clj-time.format/unparse
                   datetime-format (clj-time.core/now))
        filename (if (str/ends-with? raw-filename ".csv")
                   raw-filename
                   (format "%s.csv" raw-filename))]
    (str temp-dir timestamp "_" filename)))

(defn overwrite-csv [raw-filename data]
  {:pre [(s/valid? (s/coll-of map?) data)
         (apply = (map keys data))]}
  (let [filename (process-filename raw-filename)
        temp-filename (process-temp-filename raw-filename)]
    ;; Make sure the original file exists before trying to copy it.
    (when (.exists (io/file filename))
      (io/copy (io/file filename) (io/file temp-filename)))
    (with-open [writer (io/writer filename)]
      (csv/write-csv writer (list (->> data
                                       first
                                       keys
                                       (map keyword-to-string))))
      (csv/write-csv writer (map vals data)))
    (format "Overwrote and wrote %d rows to %s. Also saved a copy to %s"
            (count data) filename temp-filename)))

(defn append-csv [raw-filename data]
  {:pre [(s/valid? (s/coll-of map?) data)
         (every? keyword? (keys (first data)))
         (apply = (map keys data))]}
  (let [filename (process-filename raw-filename)
        existing-data (read-csv raw-filename)
        existing-header (->> existing-data first keys (map keyword-to-string))
        new-header (->> data first keys (map keyword-to-string))]
    (assert (= existing-header new-header)
            "Headers do not match!")
    (with-open [writer (io/writer filename :append true)]
      (csv/write-csv writer (map vals data)))
    (let [total-rows (count (read-csv raw-filename))]
      (format "appended %d rows to %s, total %d rows now"
              (count data) filename total-rows))))

(defn read-csv [raw-filename]
  (let [filename (process-filename raw-filename)]
    (with-open [reader (io/reader filename)]
      (let [lines (vec (csv/read-csv reader))
            header (first lines)
            rows (rest lines)]
        (->> rows
             (map (fn [row] (zipmap header row)))
             (map walk/keywordize-keys))))))
