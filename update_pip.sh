git fetch 
git pull

rm -r build
rm -r dist

python3 pipUpdateVersionNumber.py

python3 setup.py sdist bdist_wheel

twine upload dist/*
