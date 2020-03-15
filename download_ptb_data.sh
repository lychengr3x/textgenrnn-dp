# the Penn Tree Bank data first (download from Tomas Mikolov's webpage)
mkdir data

wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

tar -xf  simple-examples.tgz

mv simple-examples/data/ptb.train.txt data/
mv simple-examples/data/ptb.valid.txt data/
mv simple-examples/data/ptb.test.txt data/

rm -rf simple_examples