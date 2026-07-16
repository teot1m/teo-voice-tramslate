# Плагины UVT

Любой `.py`-файл в этом каталоге загружается при старте (ТЗ §15). Плагин
наследует интерфейс из `uvt.interfaces`, регистрируется декоратором — и сразу
доступен в конфиге и GUI. Ядро приложения менять не нужно.

Виды движков: `capture`, `vad`, `stt`, `translation`, `tts`.

## Пример: свой переводчик

```python
# plugins/my_translator.py
from uvt.interfaces import TranslationEngine
from uvt.registry import register


@register("translation", "my-translator")
class MyTranslator(TranslationEngine):
    async def warmup(self):
        ...  # загрузка клиента/модели; параметры — в self.cfg (секция translation)

    async def translate(self, text, source_lang, target_lang, context):
        return my_api_call(text, target_lang)
```

Подключение в профиле:

```yaml
translation:
  engine: my-translator
  my_custom_option: 42   # свои ключи разрешены, придут в self.cfg
```

⚠️ Плагины исполняются как обычный Python-код — кладите сюда только то, чему
доверяете.
