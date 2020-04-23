#!/bin/bash

set -eu

rm -rf .git
cd docs
make html
cd _build/html

git init

git config user.email "minami.polly@gmail.com"
git config user.name "Travis-CI"

[ -z $GITHUB_TOKEN ] && echo GITHUB_TOKEN is empty!!

git add .
git commit -m "[ci skip] Update docs"
git push --quit --force https://${GITHUB_TOKEN}@github.com/chlorochrule/cknn.git master:gh-pages >/dev/null 2>&1
