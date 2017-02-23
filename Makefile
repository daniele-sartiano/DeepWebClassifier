UTILS=utils
DOWNLOAD_DIR = /data/fine/fine-categorization/data/training

TRAINSET = /srv/fine/fine-categorization/training/domains.totrain

PROCESSES=12

fine.txt: /srv/fine/fine-categorization/training/domains.totrain
	 $(UTILS)/createDataset.py -d $(DOWNLOAD_DIR) < $< > $@

test.txt: test
	head -1000 test | $(UTILS)/createDataset.py -d $(DOWNLOAD_DIR) -p $(PROCESSES) > $@