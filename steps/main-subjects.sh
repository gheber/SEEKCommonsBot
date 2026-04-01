#!/bin/bash
set -e

# Build QID list from stdin (one QID per line).
mapfile -t qids
clean_qids=()
for qid in "${qids[@]}"; do
  qid=${qid//[[:space:]]/}
  if [ -n "$qid" ]; then
    clean_qids+=("$qid")
  fi
done

if [ ${#clean_qids[@]} -eq 0 ]; then
  echo "No QIDs provided on stdin." >&2
  exit 1
fi

tmp_out=$(mktemp)
trap 'rm -f "$tmp_out" main-subjects.rql' EXIT

# Query the main WDQS host so we can combine works in the main graph with works
# in the scholarly subgraph and return their main subjects (P921).
endpoint="https://query.wikidata.org/sparql"

for ((i=0; i<${#clean_qids[@]}; i+=100)); do
  batch=("${clean_qids[@]:i:100}")
  population_ids=$(printf 'wd:%s ' "${batch[@]}")

  cat > main-subjects.rql <<SPARQL
SELECT ?mainSubject
WHERE
{
  VALUES ?publication { $population_ids }

  {
    ?publication wdt:P921 ?mainSubject .
  }
  UNION
  {
    SERVICE wdsubgraph:scholarly_articles {
      ?publication wdt:P921 ?mainSubject .
    }
  }
}
SPARQL

  curl -s -G "$endpoint" \
    -H 'Accept: text/tab-separated-values' \
    --data-urlencode query@main-subjects.rql \
    | tail -n +2 | tr -d '"' | sed -E 's#^.*/##; s/[<>]//g' >> "$tmp_out"

  # Pause between batches to avoid hammering the endpoint.
  if [ $((i + 100)) -lt ${#clean_qids[@]} ]; then
    sleep 1
  fi
done

sort -u "$tmp_out"
