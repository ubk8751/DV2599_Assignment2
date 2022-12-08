zip:
	@rm -rf assignment
	@mkdir assignment
	@cp *.pdf README.md main.py data.py models.py validations.py spambase.csv ./assignment
	@zip assignment.zip assignment/*
	@rm -r assignment

unzip: zip
	@unzip assignment.zip

clean:
	@rm -rf assignment assignment.zip