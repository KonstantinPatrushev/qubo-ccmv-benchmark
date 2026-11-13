# Метка времени замороженного плана анализа (RFC 3161)

План анализа `docs/ANALYSIS_PLAN.md` v2.4 заморожен 2026-10-08: текст — коммит `cf5560b`,
отметка `STATUS: FROZEN` — коммит `17c9c16`. Публичная метка времени поставлена 2026-10-11 по
решению пользователя через бесплатный сервис меток времени freetsa.org (стандарт RFC 3161) вместо
OSF/Zenodo (§12 плана). Сервису отправлен только SHA-256 файла, текст плана не отправлялся.

| Что | Значение |
|---|---|
| Файл | `ANALYSIS_PLAN_v2.4_frozen_17c9c16.md` (байт в байт `git show 17c9c16:docs/ANALYSIS_PLAN.md`) |
| SHA-256 | `d9788a7310d6a1d8dcb5979b6a56ea457770c651beb76444be4d879cb1fd70ca` |
| Запрос | `ANALYSIS_PLAN_v2.4_frozen.tsq` (SHA-256 файла, nonce `0x092130CA6ADB67BA`, `-cert`) |
| Ответ (токен) | `ANALYSIS_PLAN_v2.4_frozen.tsr` |
| Время метки | 2026-10-11 14:57:09 UTC |
| Серийный номер | `0x091CFE27` |
| Служба | freetsa.org (`https://freetsa.org/tsr`), политика `tsa_policy1` |

## Как проверить

```bash
git show 17c9c16:docs/ANALYSIS_PLAN.md | sha256sum      # d9788a73…fd70ca
openssl ts -reply -in ANALYSIS_PLAN_v2.4_frozen.tsr -text   # время, хеш, серийный номер
# полная проверка подписи: корневой и TSA-сертификаты freetsa.org (https://freetsa.org/files/)
openssl ts -verify -in ANALYSIS_PLAN_v2.4_frozen.tsr -queryfile ANALYSIS_PLAN_v2.4_frozen.tsq \
    -CAfile cacert.pem -untrusted tsa.crt
```
