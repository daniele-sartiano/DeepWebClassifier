UTILS = utils
DATA = data

TRAINING_DIR = /data/fine/fine-categorization/data/training
TRAINSET = /srv/fine/fine-categorization/training/domains.totrain

PROCESSES=12

$(DATA)/fine.txt: /srv/fine/fine-categorization/training/domains.totrain
	 $(UTILS)/createDataset.py -d $(TRAINING_DIR) < $< > $@


model: $(DATA)/fine.txt
	./classifier.py

test.txt: test
	head -1000 test | $(UTILS)/createDataset.py -d $(TRAINING_DIR) -p $(PROCESSES) > $@


DOWNLOADER_DIR = /data/download

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

$(DATA)/w2v_domains.txt: $(DATA)/all_domains.txt
	$(UTILS)/selectDomains.py < $< > $@

$(DATA)/w2v_corpus.txt: $(DATA)/w2v_domains.txt
	 $(UTILS)/createDataset.py -d $(DOWNLOADER_DIR) < $< > $@
