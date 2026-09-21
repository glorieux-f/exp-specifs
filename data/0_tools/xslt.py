#!/usr/bin/env python3

import argparse
import glob
import sys
from pathlib import Path

from lxml import etree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xslt", type=Path)
    parser.add_argument("glob")
    # parser.add_argument("output", type=Path)
    # parser.add_argument("--ext", default=".xml")
    args = parser.parse_args()
    outdir = Path("data/md").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # args.output.mkdir(parents=True, exist_ok=True)

    try:
        transform = etree.XSLT(etree.parse(args.xslt))
    except (etree.XMLSyntaxError, etree.XSLTParseError) as e:
        print(f"Cannot compile XSLT {args.xslt}:", file=sys.stderr)
        for error in e.error_log:
            print(
                f"  {error.filename}:{error.line}:{error.column}: "
                f"{error.level_name}: {error.message}",
                file=sys.stderr,
            )
        raise

    for src_name in sorted(glob.glob(args.glob, recursive=True)):
        src = Path(src_name)
        print(f"### {src}", file=sys.stderr)
        doc = etree.parse(src)
        try:
            result = transform(
                doc,
                filename=etree.XSLT.strparam(src.stem),
                outdir=etree.XSLT.strparam(outdir.as_uri() + "/"),
            )
        except etree.XSLTApplyError as e:
            print(f"XSLT error while processing {src}:", file=sys.stderr)
            for error in transform.error_log:
                print(
                    f"  {error.filename}:{error.line}:{error.column}: "
                    f"{error.level_name}: {error.message}",
                    file=sys.stderr,
                )
            raise
        result.write_output(sys.stdout.buffer)
        """
        dst = args.output / src.with_suffix(args.ext).name

        result = transform(doc)
        result.write_output(dst)

        print(f"{src} -> {dst}")
        """


if __name__ == "__main__":
    main()