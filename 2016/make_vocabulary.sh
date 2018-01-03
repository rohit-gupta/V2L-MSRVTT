sed -e 's/<[^>]*>//g' msrvtt_descriptions.csv | awk -F',' '{print $2}' | tr " " "\n" | sed '/^$/d' | sort | uniq -ci | sed 's/^ *//g' | sort -Vr | sed 's/ /,/g' > vocabulary_clean.txt
