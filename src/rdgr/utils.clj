(ns rdgr.utils)

(defn in?
  "true if coll contains elm"
  [coll elm]
  {:pre [(coll? coll)]}
  (some #(= elm %) coll))
