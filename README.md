# qiuqiu-document-workshop

秋秋证件工坊

本地处理身份证正反面照片：自动检测证件边缘、透视矫正，并按身份证标准尺寸排版到一张 A4 PDF。

## 在线体验

GitHub Pages 版本：<https://qqhkx2027.github.io/qiuqiu-document-workshop/>

在线版使用浏览器本地处理，不需要启动 Python 服务；首次打开会加载 OpenCV.js 和 PDF 引擎。

## 特点

- 支持电脑选择照片，也支持手机浏览器直接调用相机
- 自动识别四角并透视矫正
- 正反面各按 `85.60 × 54.00 mm` 排版
- 输出 A4 纵向 PDF，适合打印或提交
- 原始照片只在本机内存中处理，不上传云端

## 启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

然后打开 <http://127.0.0.1:8765>。

macOS 也可以双击 `启动证件裁剪工具.command`。

GitHub Pages 部署使用 `docs/` 静态版本；本地 Python 版本保留在项目根目录，便于离线使用。

## 拍摄建议

让证件四条边完整出现在画面中，尽量保持平行，避免反光、阴影和手指遮挡。自动识别失败时重新拍摄通常比裁剪过度的照片更容易得到准确结果。

## 输出位置

生成的 PDF 保存到 `output/pdf/`。原始上传照片不会保存到项目目录。
