#!/bin/bash
> ${2}
while read line || [ "$line" ]; do
    IFS='|' read -ra arr <<< "$line"
    b="$(echo "${arr[1]}" | wc -w)"
    if [ $b -lt 19 ]
    then
        echo "$line" >> ${2}
    fi
done < "$1"

