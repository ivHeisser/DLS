В данной директории содержатся вспомогательные python-скрипты, которые не имеют прямого отношения к ML pipeline и носят вспомогательный характер. Больше относятся к разработке самого фреймворка.

# Добавление нового submodule во фреймворк

Добавление и регистрация нового submodule происходит с помощью `scripts/create_submodule.py`.

#### Использование
`python scripts/create_submodule.py symmetries`
или
`python scripts/create_submodule.py Symmetries`
или
`python scripts/create_submodule.py SYMMETRIES`

Во всех случаях получится одинаковый результат:
```
src/
└── geomml/
    ├── registry.py
    └── symmetries/
        └── __init__.py
```
В `registry.py` автоматически добавится
```
SYMMETRIES = Registry("symmetry")
```
и создастся `symmetries/__init__.py`, который будет автоматически регистрировать рython-модули в этой директории.

# Удаление существующего submodule из фреймворка

CLI строка:

`python scripts/delete_submodule.py symmetries`