#!/usr/bin/env python
import logging
import argparse
import subprocess
import os
class ReportAnalyzer:
    """
    analyze entropy_report generated by ModelXRay
    """
    def __init__(self, entropy_report, args):
        self._er = entropy_report
        if args.filter_encrypted is True:
            self._res = entropy_report + '.encrypted.filtered'
        else:
            self._res = entropy_report + '.filtered'
        self._rp = open(entropy_report,'r').readlines()
        self._args = args
        pass
    def get_apk_path(self, pkg_name):
        shell_cmd = "ag -g %s %s" % (pkg_name, self._args.apk_path)
        res = self.run_w(shell_cmd)
        return res
    def filter_encrypt(self, line):
        fields = line.split()
        ent = fields[0]
        if '7.99' in ent:
            return True
        else:
            return False

    def filter_filesz(self, line):
        fields = line.split()
        size = fields[2]
        if size.endswith('K'):
            #print("size:"+size)
            f = float(size[:-1]) # get ride of 'K' or 'M'
            if f < 5.0:
                return False
            else:
                return True
        else:
            return True
    def filter_other(self, line):
        fields = line.split()
        if len(fields) == 6:
            if fields[5].split(':')[0] == 'other':
                return True 
            else:
                return False 
        else:
            return False
    def filter_fw(self, line):
        # rule1: if no framework or only 'other' fraemwork, only keep model keywords
        fields = line.split()
        # whitelist
        if 'AliNN' in line:
            return True

        # check framework
        if len(fields) <= 5:
            return False
        elif len(fields) == 6:
            if fields[5].split(':')[0] == 'other':
                return False
            else:
                return True
        else:
            return True
    def filter_filetype(self, line):
        # filter out known non-model types like txt,lua,
        # if endswith mp3, only when path contains model keywords
        fields = line.split()
        fn = fields[4]
        ends = fn.split('.')[-1]
        if ends in ['Ew_inv','npz','qmltypes','b3d','x','blend','ffa','ffm','skel','nmf','so''Ew_inv','ExportJson','Ew','sqlite3','webp','irf','csp','pts','config','js','dll','assetbundle','db','bytes','is','often','U','g3dj','xd','tga','fs','is','yaml','manis','bf2','fbx','yml','emd','html','lm','dpm','cade','md']:
            return False
        if fn.endswith('.mp3') or fn.endswith('.mp4') or fn.endswith('.zip') or fn.endswith('.zzip'):
            if 'model' not in fn:
                return False
            else:
                return True
        elif fn.endswith('.lua') or fn.endswith('.txt') or fn.endswith('.arf') or fn.endswith('.obj'):
            return False
        else:
            return True
    def select_other_framework(self):
        ori = self._rp

        other_pass = filter(self.filter_other, ori)
        other_pkg_num = self.extract_unique_pkgname(other_pass)

        print("pure unique other framework apps:" + str(other_pkg_num))

    def filter_out_non_ml(self):
        vendor_name = os.path.basename(self._er)[:-14]

        ori = self._rp[1:]
        ori_pkg_num = self.extract_unique_pkgname(ori)

        fw_pass = filter(self.filter_fw, ori)
        fw_pkg_num = self.extract_unique_pkgname(fw_pass)

        sz_pass = filter(self.filter_filesz, fw_pass)
        sz_pkg_num = self.extract_unique_pkgname(sz_pass)

        tp_pass = filter(self.filter_filetype, sz_pass)
        tp_pkg_num = self.extract_unique_pkgname(tp_pass)

        if self._args.filter_encrypted is True:
            en_pass = filter(self.filter_encrypt, tp_pass)
            en_pkg_num = self.extract_unique_pkgname(en_pass)
            res_pass = en_pass
        else:
            en_pass = []
            en_pkg_num = []
            res_pass = tp_pass

            
        logging.info("model files vendor:%s original: %d, after fw_pass:%d, after sz_pass:%d, after tp_pass:%d en_pass:%d" %(vendor_name, len(ori), len(fw_pass), len(sz_pass), len(tp_pass), len(en_pass)))
        logging.info("app numbers vendor:%s original: %d, after fw_pass:%d, after sz_pass:%d, after tp_pass:%d en_pass:%d" %(vendor_name, len(ori_pkg_num), len(fw_pkg_num), len(sz_pkg_num), len(tp_pkg_num), len(en_pkg_num)))
        print("model files vendor:%s original: %d, after fw_pass:%d, after sz_pass:%d, after tp_pass:%d en_pass:%d" %(vendor_name, len(ori), len(fw_pass), len(sz_pass), len(tp_pass), len(en_pass)))
        print("app numbers vendor:%s original: %d, after fw_pass:%d, after sz_pass:%d, after tp_pass:%d en_pass:%d" %(vendor_name, len(ori_pkg_num), len(fw_pkg_num), len(sz_pkg_num), len(tp_pkg_num), len(en_pkg_num)))
        wh = open(self._res, 'w')
        wh.write(self._rp[0])
        wh.write(''.join(res_pass))
        pass

    def print_unique_pkgname(self, pkgnames):
        for p in pkgnames:
            if self._args.search_apkpath is True:
                ap = self.get_apk_path(p)
                logging.debug("%s:%s"%(p,ap))
                print("%s:%s"%(p,ap))
            else:
                logging.debug(p)
                print(p)
        pass
    def extract_unique_pkgname(self, report):
        pkgnames = []
        for line in report[1:]:
            fields = line.split()

            if len(fields) <=4:
                continue
            # filter non-encryption app
            #ent = fields[0]
            #if '7.99' not in ent:
            #    continue

            pkgname = fields[3]
            if pkgname == 'pkgname':
                continue
            elif pkgname in pkgnames:
                continue
            else:
                pkgnames.append(pkgname)
        return pkgnames

    def extract_unique_libraries(self):
        if self._rp[0].split()[0] == 'entropy':
            report = self._rp[1:]
        else:
            report = self._rp

        unilibs = {} 
        for line in report:
            fields = line.split()
            if len(fields) < 6:
                continue
            else:
                fwlibs = fields[5:]

                for fwlib in fwlibs:
                    if ':' not in fwlib:
                        continue
                    libs = fwlib.split(':')[1].split(',')
                    for lib in libs:
                        if lib in unilibs:
                            applist = unilibs[lib]
                            if fields[3] in applist:
                                continue
                            else:
                                applist.append(fields[3])
                        else:
                            applist = [fields[3]]
                            unilibs[lib] = applist
                    
        return unilibs
    def printlibdic(self, libdic):
        for lib in libdic:
            print("%s,%d,%s"%(lib, len(libdic[lib]), ','.join(libdic[lib])))

    def run_wo(self, shell_cmd):
        """
        run shell cmds without result returned
        """
        logging.debug("executing shell cmd : " + shell_cmd)
        res = subprocess.call(shell_cmd, shell=True)
        if res != 0:
            logging.error("error in executing cmd :" + shell_cmd)
        pass
    def run_w(self, shell_cmd):
        """
        run shell cmds with result returned
        """
        logging.debug("executing shell cmd : " + shell_cmd)
        try:
            res = os.popen(shell_cmd).read().strip()
        except:
            logging.error("error in executing : " + shell_cmd)
            res = "" 
        return res
    def analyze(self):
        if self._args.package_name is True:
            pkgnames = self.extract_unique_pkgname(self._rp[1:])
            self.print_unique_pkgname(pkgnames)
        if self._args.filter_out is True:
            self.filter_out_non_ml()
        if self._args.filter_other is True:
            self.select_other_framework()
        if self._args.unique_libraries is True:
            unilibs = self.extract_unique_libraries()
            self.printlibdic(unilibs)
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='reportanalyzer')
    parser.add_argument('entropy_report',
            help = 'path to entropy_report')
    parser.add_argument('-p', '--package-name', action='store_true',
            help = 'extract unique package name')
    parser.add_argument('-a', '--apk-path', default = '/home/ruimin/nfs/MobileDL/data/raw_apks/',
            help = 'extract unique package name')
    parser.add_argument('-f', '--filter-out', action='store_true',
            help = 'use file size, framework name, file type to filter out non-model files')
    parser.add_argument('-s', '--search-apkpath', action='store_true',
            help = 'search for apk path on server')
    parser.add_argument('-e', '--filter-encrypted', action='store_true',
            help = 'apply encryption filter')
    parser.add_argument('-o', '--filter-other', action='store_true',
            help = 'apply other filter to select apps that only has other framework')
    parser.add_argument('-l', '--unique-libraries', action='store_true',
            help = 'extract unique libraries')

    args = parser.parse_args()
    logging.basicConfig(filename='reportanalyzer.log', level=logging.INFO)
    RA = ReportAnalyzer(args.entropy_report, args)
    RA.analyze()