#!/bin/bash

set -eu

rm -rf .git
cd docs
make html
cd _build/html

git config user.email "minami.polly@gmail.com"
git config user.name "Travis-CI"

git init
git add .
git commit -m "[ci skip] Update docs"
git push --quit --force https://${GITHUB_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git master:gh-pages >/dev/null 2>&1
