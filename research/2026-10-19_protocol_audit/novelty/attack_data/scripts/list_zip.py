import sys
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
from httpzip import open_remote
z, f = open_remote()
out = open('/Users/jeongjihoon/violin-rne/novelty/attack_data/meta/zip_listing.tsv', 'w')
for i in z.infolist():
    out.write(f"{i.filename}\t{i.file_size}\t{i.compress_size}\t{i.compress_type}\t{i.header_offset}\n")
out.close()
print(len(z.infolist()), 'members; requests', f.nreq, 'bytes', f.nbytes)
