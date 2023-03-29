import os
import os.path as op
import gc
import json
from typing import List
import math
import logging
import argparse
from tqdm import tqdm
try:
    from .blob_storage import BlobStorage, disk_usage
except:
    class BlobStorage:
        pass


def generate_lineidx(filein: str, idxout: str) -> None:
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp, 'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos) + "\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)


def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)


class TSVFile(object):
    def __init__(self,
                 tsv_file: str,
                 if_generate_lineidx: bool = False,
                 lineidx: str = None,
                 class_selector: List[str] = None,
                 blob_storage: BlobStorage = None):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx' \
            if not lineidx else lineidx
        self.linelist = op.splitext(tsv_file)[0] + '.linelist'
        self.chunks = op.splitext(tsv_file)[0] + '.chunks'
        self._fp = None
        self._lineidx = None
        self._sample_indices = None
        self._class_boundaries = None
        self._class_selector = class_selector
        self._blob_storage = blob_storage
        self._len = None
        # the process always keeps the process which opens the file.
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and if_generate_lineidx:
            generate_lineidx(self.tsv_file, self.lineidx)

    def __del__(self):
        self.gcidx()
        if self._fp:
            self._fp.close()
            # physically remove the tsv file if it is retrieved by BlobStorage
            if self._blob_storage and 'azcopy' in self.tsv_file and os.path.exists(self.tsv_file):
                try:
                    original_usage = disk_usage('/')
                    os.remove(self.tsv_file)
                    logging.info("Purged %s (disk usage: %.2f%% => %.2f%%)" %
                                 (self.tsv_file, original_usage, disk_usage('/') * 100))
                except:
                    # Known issue: multiple threads attempting to delete the file will raise a FileNotFound error.
                    # TODO: try Threadling.Lock to better handle the race condition
                    pass

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def gcidx(self):
        logging.debug('Run gc collect')
        self._lineidx = None
        self._sample_indices = None
        #self._class_boundaries = None
        return gc.collect()

    def get_class_boundaries(self):
        return self._class_boundaries

    def num_rows(self, gcf=False):
        if (self._len is None):
            self._ensure_lineidx_loaded()
            retval = len(self._sample_indices)

            if (gcf):
                self.gcidx()

            self._len = retval

        return self._len

    def seek(self, idx: int):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[self._sample_indices[idx]]
        except:
            logging.info('=> {}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx: int):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def get_key(self, idx: int):
        return self.seek_first_column(idx)

    def __getitem__(self, index: int):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logging.debug('=> loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                lines = fp.readlines()
                lines = [line.strip() for line in lines]
                self._lineidx = [int(line) for line in lines]

            # read the line list if exists
            linelist = None
            if op.isfile(self.linelist):
                with open(self.linelist, 'r') as fp:
                    linelist = sorted(
                        [
                            int(line.strip())
                            for line in fp.readlines()
                        ]
                    )

            if op.isfile(self.chunks):
                self._sample_indices = []
                self._class_boundaries = []
                class_boundaries = json.load(open(self.chunks, 'r'))
                for class_name, boundary in class_boundaries.items():
                    start = len(self._sample_indices)
                    if class_name in self._class_selector:
                        for idx in range(boundary[0], boundary[1] + 1):
                            # NOTE: potentially slow when linelist is long, try to speed it up
                            if linelist and idx not in linelist:
                                continue
                            self._sample_indices.append(idx)
                    end = len(self._sample_indices)
                    self._class_boundaries.append((start, end))
            else:
                if linelist:
                    self._sample_indices = linelist
                else:
                    self._sample_indices = list(range(len(self._lineidx)))

    def _ensure_tsv_opened(self):
        if self._fp is None:
            if self._blob_storage:
                self._fp = self._blob_storage.open(self.tsv_file)
            else:
                self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.debug('=> re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()


class TSVWriter(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
        self.tsv_file_tmp = self.tsv_file + '.tmp'
        self.lineidx_file_tmp = self.lineidx_file + '.tmp'

        self.tsv_fp = open(self.tsv_file_tmp, 'w')
        self.lineidx_fp = open(self.lineidx_file_tmp, 'w')

        self.idx = 0

    def write(self, values, sep='\t'):
        v = '{0}\n'.format(sep.join(map(str, values)))
        self.tsv_fp.write(v)
        self.lineidx_fp.write(str(self.idx) + '\n')
        self.idx = self.idx + len(v)

    def close(self):
        self.tsv_fp.close()
        self.lineidx_fp.close()
        os.rename(self.tsv_file_tmp, self.tsv_file)
        os.rename(self.lineidx_file_tmp, self.lineidx_file)




# ----------------------------------------------------------------- # 


def split_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]




def split(tsv_path, lineidx_path, out_folder, keep_folder_if_exist, chuck_size):
    
    reader = TSVFile(tsv_file=tsv_path, lineidx=lineidx_path)

    N = len(reader)

    file_size = os.path.getsize(tsv_path)
    num_splits = math.ceil(N / (file_size // chuck_size))

    print("TSV file size (bytes): ", file_size)
    print("TSV file row count: ", N)
    print("TSV file chunk size (bytes): ", chuck_size)
    print("TSV file chunk count (entries): ", num_splits)

    idx_range_list = list(split_chunks( list(range(N)), num_splits))
    print("TSV sub file count (splits): ", len(idx_range_list))

    if os.path.isdir(out_folder) and (not keep_folder_if_exist):
        import shutil
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)


    for tsv_file_idx, idx_range in tqdm(enumerate(idx_range_list), total=len(idx_range_list)):

        tsv_file = f'train-{tsv_file_idx:04d}.tsv'
        tsv_file = os.path.join( out_folder,  tsv_file)

        writer = TSVWriter(tsv_file)
        for idx in idx_range:
            row = reader[idx]
            writer.write(row)
        writer.close()



def merge(merge_in_folder, merge_out_folder):

    files = os.listdir(merge_in_folder)
    N = len(files)

    lineidx_files = files[0:N//2]
    tsv_files = files[N//2:N]


    if os.path.isdir(merge_out_folder):
        import shutil
        shutil.rmtree(merge_out_folder)
    os.makedirs(merge_out_folder, exist_ok=True)
    writer = TSVWriter( os.path.join(merge_out_folder,'train-00.tsv') )


    for tsv_file, lineidx_file in tqdm(zip(tsv_files, lineidx_files)):
        
        tsv_file = os.path.join(merge_in_folder, tsv_file)
        lineidx_file = os.path.join(merge_in_folder, lineidx_file)

        reader = TSVFile(tsv_file=tsv_file, lineidx=lineidx_file)
        for idx in range(len(reader)):
            row = reader[idx]
            writer.write(row)
    
    writer.close()



if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    # split args 
    parser.add_argument("--tsv_path", type=str,  default='train-00.tsv', help="")
    parser.add_argument("--lineidx_path", type=str,  default='train-00.lineidx', help="")
    parser.add_argument("--out_folder", type=str,  default='out', help="")
    parser.add_argument('--keep_folder_if_exist', action='store_true') 
    parser.add_argument('--chuck_size', type=int, default=1024*1024*1024)
    
    
    # merge args
    parser.add_argument("--merge_in_folder", type=str,  default='out', help="")
    parser.add_argument("--merge_out_folder", type=str,  default='merged_tsv', help="")

    args = parser.parse_args()

    # split(
    #     tsv_path = args.tsv_path,
    #     lineidx_path = args.lineidx_path,
    #     out_folder = args.out_folder,
    #     keep_folder_if_exist = args.keep_folder_if_exist,
    #     chuck_size = args.chuck_size,
    # )


    merge(args.merge_in_folder, args.merge_out_folder)


