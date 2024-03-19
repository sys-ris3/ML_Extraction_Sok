#!/bin/sh
for i in {1..13}
do
    mv post_dwconv"$i"_layers.cfg dwconv"$i".cfg
    mv post_pwconv"$i"_layers.cfg pwconv"$i".cfg
done
