## 472

## helsinki
python -u test.py -l es -m helsinki -tf data/expanded-test-472-full.json -gf generations/expanded-test-472-full.es.txt
python -u test.py -l fr -m helsinki -tf data/expanded-test-472-full.json -gf generations/expanded-test-472-full.fr.txt

## m2m100
python -u test.py -l zh -m m2m100 -tf data/expanded-test-472-full.json -gf generations/expanded-test-472-full.zh.txt
python -u test.py -l ru -m m2m100 -tf data/expanded-test-472-full.json -gf generations/expanded-test-472-full.ru.txt

## nllb200
python -u test.py -l en -m nllb200 -tf data/expanded-test-472-full.json -gf generations/expanded-test-472-full.en.txt

## WEBNLG

## helsinki
python -u test.py -l br -m helsinki -tf data/expanded-test-webnlg-full.json -gf generations/expanded-test-webnlg-full.br.txt
python -u test.py -l es -m helsinki -tf data/expanded-test-webnlg-full.json -gf generations/expanded-test-webnlg-full.es.txt
python -u test.py -l mt -m helsinki -tf data/expanded-test-webnlg-full.json -gf generations/expanded-test-webnlg-full.mt.txt

## m2m100
python -u test.py -l cy -m m2m100 -tf data/expanded-test-webnlg-full.json -gf generations/expanded-test-webnlg-full.cy.txt
python -u test.py -l ru -m m2m100 -tf data/expanded-test-webnlg-russian.json -gf generations/expanded-test-webnlg-russian.ru.txt

## nllb200
python -u test.py -l en -m nllb200 -tf data/expanded-test-webnlg-full.json -gf generations/expanded-test-webnlg-full.en.txt
python -u test.py -l ga -m nllb200 -tf data/expanded-test-webnlg-full.json -gf generations/expanded-test-webnlg-full.ga.txt
