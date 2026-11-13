 
# Tapaaminen 2026-11-03


## Valmistautuminen

- Testaillaan Codexia
- Katsotaan tooleja ja kirjastoja (esim. `uv` ja `make`)
- Tavataan testihackathonaamaan ens ti klo 16 (2026-11-10)


## Software stack ja setuppaus


### Git-tietosäilöt

Pääsäilö osoitteessa (git@github.com:joelkronqvist/junction26.git).

Backup-säilö löytyy (ssh://git-junction@cron4.fi:60761/srv/junction26.git).

Käytetään feature brancheja.


### Kielet

Build tool: `make` tarvittaessa

- Python
  - Kirjastot: numpy, matplotlib, scipy
  - CLI: argparse, readline
  - Sklearn, tai PyTorch jos SKLearn on liian intensiivinen
  - API backend: esim. FastAPI, starlette, ...
  - Build tool: `uv`
  - Tyyppitarkistin: Pyright
  - Formatointi: `ruff`
  - Testaus: unittest + doctest-esimerkit
  - Dokumentointi: docstringit

### Kirjastot

- CLI:
- Backend:
- Frontend:
- Loggaus:

- Testaus:
- Esitys / diat / ym: Powerpoint, matplotlib
  - Kuvaajat: matplotlib
  
      
### AI:n käyttö

- [ ] Lisää AGENTS.md (Sampsa)
  - Ylläpidetään dokumentaatiota AI:n kontekstiksi
- subdirectory AI-slopille
  - Kaikki ei-slop-directory koodi jotakuinkin hyvää ja ymmärrettyä: voisit itse seistä koodin valintojen takana
- AI:n käytön dokumentointi
  - Täydet lokit mikäli mahdollista (repon ulkopuolelle)
  - Tiivistelmät jokaisesta sessiosta => Lopullinen tiivistelmä


## Lepokäytännöt



