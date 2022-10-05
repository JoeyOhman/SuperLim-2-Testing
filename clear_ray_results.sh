#!/usr/bin/bash
echo "Clearing ray_results checkpoints!"
for i in $(find results/ray_results/ -type f| \grep -v "tfevents"); do rm -f $i; done
find results/ray_results/ -type d -empty -delete

echo "Clearing trainer checkpoints!"
for i in $(find results/run-* -type f| \grep -v "tfevents"); do rm -f $i; done
find results/run-* -type d -empty -delete
