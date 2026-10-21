### Neural Network

- [ ]  diskujson lokal VS globale modeller VS finetuned
- [ ]  featureengineering
    - [ ]  teste med å legge til features ved å se på om de nye featurene er korrelert med output
    - [ ]  vis at noen engineered features ikke er viktige fordi de ikke er korrolert med consumption
- [ ]  kilder for hvilke modeler er best
- [ ]  kilder for valg av baseline, evt en god begrunnelse
- [ ]  begrunne alle valg relativt tydelig
    - [ ]  RMSE vs MAE
    - [ ]  tydeligere argumentasjon for modellvalg
- [ ]  bygg om neural net til å bruke sliding window


---

### business:

- [ ]  KPI: få store feil er verre enn mange små, pga eksonentielle pris hopp [hør med Niklas]
    - [ ]  få frem at de få gangene vi er over en treshdol på si ±5% så koster det operatøren vår utrolig mye
- [ ]  paradigm changes: modellen vår kan ikke få med seg paradignm changes
    - [ ]  som at i 2022 var det langt færre elbiler som lader på natta enn i 2026 og at enkelthendelser som en ny utenlandskabel har massiv innvirkning på prisene
        - slike år-til-år endringer får vi ikke med oss
- [ ]  cost risk & tradeoff
    - [ ]  Tradeoff: RMSE vs MAE
    - [ ]  cost/tradeoff: training model: expanding window VS single head (ask claude). this is more complex (refering from validation task3)
- [ ]  cost & risk:
    - [ ]  overfitting from complex models; they have higher cost and maybe better paperperformance but fail to generalise