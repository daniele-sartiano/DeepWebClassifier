UTILS = utils
DATA = data

TRAINING_DIR = /data/fine/fine-categorization/data/training
TRAINSET = /data/fine/fine-categorization/training/domains.totrain /data/fine/fine-categorization/training/domains.todev

PROCESSES=12

$(DATA)/fine.txt: $(TRAINSET)
	 cat $(TRAINSET) | awk '{printf ("%s\t0\t%s\n", $$1, $$2)}' | $(UTILS)/createDataset.py -d $(TRAINING_DIR) -p 18 > $@


# Word Embeddings Section

EMBEDDINGS_SIZE=300
DOWNLOADER_DIR = /data/download
WORD2VEC = /home/daniele/fine/wordembeddings/tools/word2vec


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


$(DATA)/w2v_corpus.tok: $(DATA)/w2v_corpus.tsv
	cut -f2 < $< | $(UTILS)/normalizer.py > $@


$(DATA)/word_embeddings_$(EMBEDDINGS_SIZE).txt: $(DATA)/w2v_corpus.tok
	$(UTILS)/w2v.py --size $(EMBEDDINGS_SIZE) --filename $@ < $<


$(DATA)/word_embeddings_$(EMBEDDINGS_SIZE).w2v.txt: $(DATA)/w2v_corpus.tok
	$(WORD2VEC)/word2vec -train $< -output $@ \
           -cbow 1 -size $(EMBEDDINGS_SIZE) -window 5 -min-count 5 -negative 0 -hs 1 \
           -sample 1e-3 -threads 18 -debug 0


MAX_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000


model: $(DATA)/fine.txt $(DATA)/word_embeddings_$(EMBEDDINGS_SIZE).w2v.txt
	./classifier.py -mw $(MAX_WORDS) -msl $(MAX_SEQUENCE_LENGTH) -e $(word 2,$^) < $<

test.txt: test
	head -1000 test | $(UTILS)/createDataset.py -d $(TRAINING_DIR) -p $(PROCESSES) > $@
