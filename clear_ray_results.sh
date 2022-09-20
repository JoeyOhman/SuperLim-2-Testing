#!/usr/bin/bash
for i in $(find results/ray_results/ -type f| \grep -v "tfevents"); do rm -f $i; done
find results/ray_results/ -type d -empty -delete
