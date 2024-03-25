import torch

# 加载.pth文件
data = torch.load('corpus.pth')

# 查看文件中的数据
print("Corpus data:")
print("  - Corpus length:", len(data['corpus']))
print("  - Word to index dictionary:", data['word2ix'])
print("  - Index to word dictionary:", data['ix2word'])
print("  - Unknown token:", data['unknown'])
print("  - End of sentence token:", data['eos'])
print("  - Start of sentence token:", data['sos'])
print("  - Padding token:", data['padding'])