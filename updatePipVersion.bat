echo Opening anaconda prompt
call "C:\Users\%USERNAME%\Anaconda3\Scripts\activate.bat"

echo Pulling master branch from git
git fetch
git pull

echo Deleting build and dist folders
rmdir build /s
rmdir dist /s

python pipUpdateVersionNumber.py

echo Create new build and dist folders
python setup.py sdist bdist_wheel

echo Commit and push to git
git commit -a -m "Updated pip version to match latest commit"
git push

echo Upload to PyPi
twine upload dist/*

echo Done!

PAUSE