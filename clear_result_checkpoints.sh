#!/usr/bin/bash
echo "Clearing ray_results checkpoints!"
for i in $(find results/ray_results/ -type f| \grep -v "tfevents"); do rm -f $i; done
find results/ray_results/ -type d -empty -delete

echo "Clearing trainer_output run-*!"
for i in $(find results/trainer_output/run-* -type f| \grep -v "tfevents"); do rm -f $i; done
find results/trainer_output/run-* -type d -empty -delete

echo "Clearing trainer_output checkpoint-*!"
for i in $(find results/trainer_output/checkpoint-* -type f| \grep -v "tfevents"); do rm -f $i; done
find results/trainer_output/checkpoint-* -type d -empty -delete
