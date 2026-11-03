from modelscope import snapshot_download
snapshot_download('BAAI/bge-small-en-v1.5', local_dir='./models/bge-small-en-v1.5')
print("done")