UTILS = utils
DATA = data

TRAINING_DIR = /data/fine/fine-categorization/data/training
TRAINSET = /data/fine/fine-categorization/training/domains.totrain /data/fine/fine-categorization/training/domains.todev


TRAINING_DIR_ICT = /data/download
TRAINSET_ICT = /srv/fine/fine-categorization/training/ict_20170407/domains.totrain /srv/fine/fine-categorization/training/ict_20170407/domains.todev
TESTSET_ICT = /srv/fine/fine-categorization/training/ict_20170407/domains.totest

TESTSET = /data/fine/fine-categorization/training/domains.totest

PROCESSES=12

$(DATA)/fine.txt: $(TRAINSET)
	 cat $< | awk '{printf ("%s\t0\t%s\n", $$1, $$2)}' | $(UTILS)/createDataset.py -d $(TRAINING_DIR) -p 18 > $@


$(DATA)/fine_test.txt: $(TESTSET)
	 cat $< | awk '{printf ("%s\t0\t%s\n", $$1, $$2)}' | $(UTILS)/createDataset.py -d $(TRAINING_DIR) -p 1 > $@


$(DATA)/fine_ict.txt: $(TRAINSET_ICT)
	 cat $^ | awk '{printf ("%s\t0\t%s\n", $$1, $$2)}' | $(UTILS)/createDataset.py -d $(TRAINING_DIR_ICT) -p 18 > $@


$(DATA)/fine_ict_test.txt: $(TESTSET_ICT)
	 cat $< | awk '{printf ("%s\t0\t%s\n", $$1, $$2)}' | $(UTILS)/createDataset.py -d $(TRAINING_DIR_ICT) -p 1 > $@


# Word Embeddings Section

EMBEDDINGS_SIZE=300
DOWNLOADER_DIR = /data/download
WORD2VEC = /home/daniele/fine/wordembeddings/tools/word2vec

LOWER=_lower
P_LOWER=--lower

$(DATA)/all_domains.txt:
	for a in `ls -d $(DOWNLOADER_DIR)/*` ; do \
		for b in `ls -d $$a/*` ; do \
			for c in `ls -d $$b/*` ; do \
				for domain in `ls -d $$c/*`; do \
					echo $$domain >> $@; \
				done \
			done \
		done \
	done

$(DATA)/w2v_corpus.tsv: $(DATA)/all_domains.txt
	$(UTILS)/selectDomains.py < $< | $(UTILS)/createDataset.py -d $(DOWNLOADER_DIR) -p 12 > $@


$(DATA)/w2v_corpus$(LOWER).tok: $(DATA)/w2v_corpus.tsv
	cut -f2 < $< | $(UTILS)/normalizer.py $(P_LOWER) > $@

$(DATA)/codes$(LOWER).bpe: $(DATA)/w2v_corpus$(LOWER).tok
	head -1000000 $< | ../subword-nmt/learn_bpe.py -s 100000 > $@

$(DATA)/w2v_corpus$(LOWER).tok.bpe: $(DATA)/w2v_corpus$(LOWER).tok $(DATA)/codes$(LOWER).bpe
	 ../subword-nmt/apply_bpe.py -c $(word 2,$^) < $< > $@

$(DATA)/w2v_corpus$(LOWER).tok.windows: $(DATA)/w2v_corpus$(LOWER).tok
	 $(UTILS)/sentence2windows.py < $< > $@

$(DATA)/word_embeddings_$(EMBEDDINGS_SIZE)$(LOWER).txt.bpe: $(DATA)/w2v_corpus$(LOWER).tok.bpe
	$(UTILS)/w2v.py --size $(EMBEDDINGS_SIZE) --filename $@ < $<

$(DATA)/word_embeddings_$(EMBEDDINGS_SIZE)$(LOWER).txt: $(DATA)/w2v_corpus$(LOWER).tok
	$(UTILS)/w2v.py --size $(EMBEDDINGS_SIZE) --filename $@ < $<


VOCABULARY_SIZE=500kB

# Create vocabulary:
$(DATA)/word_embeddings-vocabulary$(LOWER).txt.gz: $(DATA)/w2v_corpus$(LOWER).tok
	cat $< | tr [:space:] '\n' | sort | uniq -c | sort -rn | gzip > $@

# </s> must be first token
$(DATA)/word_embeddings-vocabulary-$(VOCABULARY_SIZE)$(LOWER).w2v: $(DATA)/word_embeddings-vocabulary$(LOWER).txt.gz
	echo "</s> 500000" > $@
	zcat $< | head --lines=$(VOCABULARY_SIZE) | awk '{ print $$2, $$1 }' >> $@


$(DATA)/word_embeddings_$(EMBEDDINGS_SIZE)$(LOWER).w2v.txt: $(DATA)/w2v_corpus$(LOWER).tok $(DATA)/word_embeddings-vocabulary-$(VOCABULARY_SIZE)$(LOWER).w2v
	$(WORD2VEC)/word2vec -train $< -output $@ \
	   -read-vocab $(DATA)/word_embeddings-vocabulary-$(VOCABULARY_SIZE)$(LOWER).w2v \
           -cbow 1 -size $(EMBEDDINGS_SIZE) -window 5 -min-count 5 -negative 0 -hs 1 \
           -sample 1e-3 -threads 18 -debug 2

$(DATA)/word_embeddings_$(EMBEDDINGS_SIZE)$(LOWER)_no_vocab.w2v.txt: $(DATA)/w2v_corpus$(LOWER).tok
	$(WORD2VEC)/word2vec -train $< -output $@ \
           -cbow 1 -size $(EMBEDDINGS_SIZE) -window 5 -min-count 5 -negative 0 -hs 1 \
           -sample 1e-3 -threads 18 -debug 2


$(DATA)/word_embeddings_$(EMBEDDINGS_SIZE)$(LOWER).txt.windows: $(DATA)/w2v_corpus$(LOWER).tok.windows
	$(WORD2VEC)/word2vec -train $< -output $@ \
           -cbow 1 -size $(EMBEDDINGS_SIZE) -window 10 -min-count 5 -negative 0 -hs 1 \
           -sample 1e-3 -threads 18 -debug 2

MAX_WORDS = 1000000
MAX_SEQUENCE_LENGTH = 1000
MAX_SEQUENCE_LENGTH_DOMAINS = 10

BATCH=64

model$(LOWER): $(DATA)/fine.txt $(DATA)/word_embeddings_$(EMBEDDINGS_SIZE)$(LOWER).w2v.txt $(DATA)/vectors-wikipedia-lower_300.txt
	./classifier.py train --max-words-content $(MAX_WORDS) --max-sequence-length-content $(MAX_SEQUENCE_LENGTH) --embeddings $(word 2,$^) \
		--max-words-domains $(MAX_WORDS) --max-sequence-length-domains $(MAX_SEQUENCE_LENGTH_DOMAINS) \
		--embeddings-domains $(word 3,$^) --batch $(BATCH) -f $(DATA)/`date +%s`$(LOWER).model $(P_LOWER) < $<


model$(LOWER)_bpe: $(DATA)/fine.txt $(DATA)/word_embeddings_$(EMBEDDINGS_SIZE)$(LOWER).txt.bpe $(DATA)/vectors-wikipedia-lower.txt $(DATA)/codes$(LOWER).bpe
	./classifier.py train --max-words-content 200000 --max-sequence-length-content 1000 --embeddings $(word 2,$^) \
		--max-words-domains $(MAX_WORDS) --max-sequence-length-domains $(MAX_SEQUENCE_LENGTH_DOMAINS) \
		--embeddings-domains $(word 3,$^) --bpe $(word 4,$^) --batch $(BATCH) -f $(DATA)/`date +%s`_bpe$(LOWER).model $(P_LOWER) < $<


model$(LOWER)_windows: $(DATA)/fine.txt $(DATA)/word_embeddings_$(EMBEDDINGS_SIZE)$(LOWER).txt.windows $(DATA)/vectors-wikipedia-lower.txt
	./classifier.py train --max-words-content $(MAX_WORDS) --max-sequence-length-content 10000 --embeddings $(word 2,$^) \
		--max-words-domains $(MAX_WORDS) --max-sequence-length-domains $(MAX_SEQUENCE_LENGTH_DOMAINS) \
		--embeddings-domains $(word 3,$^) -w 3 --batch $(BATCH) -f $(DATA)/`date +%s`_windows$(LOWER).model $(P_LOWER) < $<


model$(LOWER)_novocab: $(DATA)/fine.txt $(DATA)/word_embeddings_$(EMBEDDINGS_SIZE)$(LOWER)_no_vocab.w2v.txt $(DATA)/vectors-wikipedia-lower.txt
	./classifier.py train --max-words-content 5000000 --max-sequence-length-content $(MAX_SEQUENCE_LENGTH) --embeddings $(word 2,$^) \
		--max-words-domains $(MAX_WORDS) --max-sequence-length-domains $(MAX_SEQUENCE_LENGTH_DOMAINS) \
		--embeddings-domains $(word 3,$^) --batch $(BATCH) -f $(DATA)/`date +%s`$(LOWER).model $(P_LOWER) < $<


model_ict$(LOWER): $(DATA)/fine_ict.txt $(DATA)/word_embeddings_$(EMBEDDINGS_SIZE)$(LOWER).w2v.txt $(DATA)/vectors-wikipedia-lower.txt
	./classifier.py train --max-words-content $(MAX_WORDS) --max-sequence-length-content $(MAX_SEQUENCE_LENGTH) --embeddings $(word 2,$^) \
		--max-words-domains $(MAX_WORDS) --max-sequence-length-domains $(MAX_SEQUENCE_LENGTH_DOMAINS) \
		--embeddings-domains $(word 3,$^) --batch $(BATCH) -f $(DATA)/`date +%s`_ict_$(LOWER).model $(P_LOWER) < $<


MODEL=

test: $(DATA)/fine_test.txt
	./classifier.py test -f $(MODEL) < $<


model.tuned: $(DATA)/fine.txt data/word_embeddings_300.tuning.txt $(DATA)/vectors-wikipedia.txt
	./classifier.py --max-words-text $(MAX_WORDS) --max-sequence-length-text $(MAX_SEQUENCE_LENGTH) --embeddings $(word 2,$^) \
		--max-words-domains $(MAX_WORDS) --max-sequence-length-domains $(MAX_SEQUENCE_LENGTH_DOMAINS) \
		--embeddings-domains $(word 3,$^) --batch $(BATCH) < $<


model.wiki: $(DATA)/fine.txt $(DATA)/vectors-wikipedia.txt
	./classifier.py train --max-words-content $(MAX_WORDS) --max-sequence-length-content $(MAX_SEQUENCE_LENGTH) --embeddings $(word 2,$^) \
		--max-words-domains $(MAX_WORDS) --max-sequence-length-domains $(MAX_SEQUENCE_LENGTH_DOMAINS) \
		--embeddings-domains $(word 2,$^) --batch $(BATCH) -f $(DATA)/wiki-`date +%s`.model < $<


test.txt: test
	head -1000 test | $(UTILS)/createDataset.py -d $(TRAINING_DIR) -p $(PROCESSES) > $@



# MAX_WORDS = 100000
# MAX_SEQUENCE_LENGTH = 20000
# MAX_SEQUENCE_LENGTH_DOMAINS = 20

# BATCH=32

# model: $(DATA)/fine.txt $(DATA)/word_embeddings_$(EMBEDDINGS_SIZE).w2v.txt $(DATA)/vectors-wikipedia.txt
# 	./classifier.py --max-words $(MAX_WORDS) --max-sequence-length $(MAX_SEQUENCE_LENGTH) --embeddings $(word 2,$^) \
# 		--max-words-domains $(MAX_WORDS) --max-sequence-length-domains $(MAX_SEQUENCE_LENGTH_DOMAINS) \
# 		--embeddings-domains $(word 3,$^) --batch $(BATCH) < $<

# model.tuned: $(DATA)/fine.txt data/word_embeddings_300.tuning.txt $(DATA)/vectors-wikipedia.txt
# 	./classifier.py --max-words $(MAX_WORDS) --max-sequence-length $(MAX_SEQUENCE_LENGTH) --embeddings $(word 2,$^) \
# 		--max-words-domains $(MAX_WORDS) --max-sequence-length-domains $(MAX_SEQUENCE_LENGTH_DOMAINS) \
# 		--embeddings-domains $(word 3,$^) --batch $(BATCH) < $<


# model.wiki: $(DATA)/fine.txt $(DATA)/vectors-wikipedia.txt
# 	./classifier.py --max-words $(MAX_WORDS) --max-sequence-length $(MAX_SEQUENCE_LENGTH) --embeddings $(word 2,$^) \
# 		--max-words-domains $(MAX_WORDS) --max-sequence-length-domains $(MAX_SEQUENCE_LENGTH_DOMAINS) \
# 		--embeddings-domains $(word 2,$^) --batch $(BATCH) < $<


# test.txt: test
# 	head -1000 test | $(UTILS)/createDataset.py -d $(TRAINING_DIR) -p $(PROCESSES) > $@
