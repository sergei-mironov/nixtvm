#!/bin/sh

find -name '*cc' -or -name '*hpp' -or -name '*h' | xargs ctags
