# Генерация текстур Minecraft автоэнкодером

Convolutional Autoencoder со skip-connections для генерации и интерполяции текстур блоков Minecraft.

## Данные

1111 текстур блоков из Minecraft (16x16 PNG).
При запуске апскейлятся до 160x160 методом Nearest Neighbor (сохраняется пиксельный стиль).

Разбиение: 90% train / 10% val.
Аугментации на train: RandomHorizontalFlip, ColorJitter.

## Архитектура ConvAE160

### Encoder

```
3x160x160 -> Conv(3->64, s2) -> BN -> ReLU  [h1: 64x80x80]
          -> Conv(64->128, s2) -> BN -> ReLU [h2: 128x40x40]
          -> Conv(128->256, s2) -> BN -> ReLU [h3: 256x20x20]
          -> Conv(256->640, s2) -> BN -> ReLU [640x10x10]
          -> Flatten -> FC(64000->2304) -> ReLU
          -> FC(2304->176) = z
```

### Decoder

```
z -> FC(176->2304) -> ReLU -> FC(2304->64000) -> ReLU -> Reshape [640x10x10]
  -> ConvT(640->256, s2) + skip h3 [256x20x20]
  -> ConvT(256->128, s2) + skip h2 [128x40x40]
  -> ConvT(128->64, s2) + skip h1 [64x80x80]
  -> ConvT(64->3, s2) -> Sigmoid [3x160x160]
```

### Двойной лосс

Каждый батч проходит две ветки:

1. **Main** (со skip-connections): `x -> encode -> decode(z, skips)` - точная реконструкция
2. **Latent-only** (без skip): `z + N(0, 0.15) -> decode(z_noisy, None)` - генерация из латента

```
L_ae = 0.3 * MSE + 0.7 * L1
L_total = 0.7 * L_main + 0.3 * L_latent
```

## Запуск

```bash
python main.py --epochs 7 --latent-dim 176 --lr 3e-4
```

## Результаты

За 7 эпох:
- train_loss: 0.176 -> 0.054
- val_loss: 0.135 -> 0.075
- val_recon: 0.120 -> 0.038

Реконструкции практически идентичны оригиналам.
Генерация вокруг латентных кодов дает новые вариации текстур с сохранением стиля.
Интерполяция через slerp дает плавные переходы между блоками.

## Выходные файлы

- `out/recon_160_epoch_*.png` - реконструкция vs оригинал
- `out/samples_160_epoch_*.png` - сгенерированные вариации
- `out/latent_interpolation_160.gif` - slerp-интерполяция (ping-pong)
- `out/losses.png` - кривые обучения