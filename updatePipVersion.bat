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

echo Upload to PyPi
twine upload dist/*

echo Done!
echo Don't forget to commit and push to git :)

PAUSE