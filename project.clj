(defproject rdgr "0.1.0-SNAPSHOT"
  :description "Code for RDGR system"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [ring/ring-core "1.8.2"]
                 [ring/ring-jetty-adapter "1.8.2"]
                 [cider/cider-nrepl "0.29.0"]
                 [org.clojure/tools.namespace "1.3.0"]
                 [compojure "1.7.0"]
                 [ring-cors "0.1.13"]
                 [ring/ring-json "0.5.1"]
                 [org.clojure/data.json "2.4.0"]
                 [io.aviso/pretty "1.3"]
                 [net.clojars.wkok/openai-clojure "0.6.0"]
                 [dev.weavejester/medley "1.6.0"]
                 [clj-time "0.15.2"]
                 [datalevin "0.8.16"]
                 [org.clojure/test.check "1.1.1"]
                 [clj-http "3.12.3"]
                 [pdfboxing/pdfboxing "0.1.15.3-SNAPSHOT"]
                 [clj-python/libpython-clj "2.024"]
                 [com.rpl/specter "1.1.4"]
                 [philoskim/debux "0.8.3"]
                 [expound "0.9.0"]
                 [adzerk/env "0.4.0"]
                 [metosin/jsonista "0.3.7"]]
  :main ^:skip-aot rdgr.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all
                       :jvm-opts ["-Dclojure.compiler.direct-linking=true"]}})
