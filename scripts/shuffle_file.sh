
echo "Shuffle file ${1}"
shuf $1 -o .suffled_temp.txt

echo "Remove unshuffled file"
rm $1

echo "Writing shuffled file to ${1}"
mv .suffled_temp.txt $1
