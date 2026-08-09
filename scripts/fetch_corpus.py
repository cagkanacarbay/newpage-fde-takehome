# /// script
# requires-python = ">=3.12"
# dependencies = ["pypdf>=5.1.0"]
# ///
"""Download the longevity research corpus used by this assistant.

The corpus is 27 open-access geroscience papers — the biology of aging, biomarkers
of biological age, and interventions that aim to extend healthspan. Life-sciences
literature was chosen because answering questions across a body of research papers
is a use case a client is likely to actually have: the documents are long, densely
technical, full of tables, and they disagree with each other.

Papers 021-027 are a second, equation-focused batch: mathematical models of aging
(Gompertz-Makeham mortality laws, reliability/network derivations, stochastic
resilience models, penalized-regression clock objectives). They exist to stress-test
whether typeset math survives PDF parsing — the "whether equations survive" question
flagged as untested in docs/parsing.html.

The PDFs are not committed to this repository. This script reproduces the corpus
byte-for-byte from its sources, so anyone cloning the repo can rebuild it.

Every paper is in the PubMed Central Open Access subset and carries a Creative
Commons licence. PDFs are fetched from Europe PMC. No paywall is bypassed.

Usage:

    uv run scripts/fetch_corpus.py                 # fetch everything that is missing
    uv run scripts/fetch_corpus.py --verify-only   # check what is already on disk
    uv run scripts/fetch_corpus.py --force         # re-download even if present
    uv run scripts/fetch_corpus.py --only 1 2 3    # fetch specific entries
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Europe PMC serves the publisher PDF from this route. The `pmc.ncbi.nlm.nih.gov`
# equivalent returns an HTML challenge page regardless of User-Agent, so it is not
# a usable fallback.
PDF_URL = "https://europepmc.org/articles/{pmcid}?pdf=render"

# Europe PMC rate-limits parallel requests and answers with HTTP 200 and a ~20-byte
# "Rate limit exceeded" body rather than a 429, so a status check alone will not
# catch throttling. Every response is size- and magic-byte-checked instead.
MIN_PDF_BYTES = 100_000
REQUEST_DELAY_SECONDS = 20.0
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 30.0
TIMEOUT_SECONDS = 120.0
USER_AGENT = "newpage-fde-corpus-fetch/1.0 (research corpus reproduction; contact via repo)"

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus" / "longevity"


@dataclass(frozen=True)
class Paper:
    """One corpus document and where it came from."""

    index: int
    slug: str
    pmcid: str
    doi: str
    title: str
    licence: str

    @property
    def filename(self) -> str:
        return f"{self.index:03d}-{self.slug}.pdf"


PAPERS: tuple[Paper, ...] = (
    Paper(1, "sanada-2025-hallmarks-of-aging-therapeutic-targets", "PMC12259695",
          "10.3389/fcvm.2025.1631578",
          "Targeting the hallmarks of aging: mechanisms and therapeutic opportunities",
          "CC BY"),
    Paper(2, "garcia-barranquero-2025-sens-vs-hallmarks-of-aging-debate", "PMC12052809",
          "10.1007/s10522-025-10248-5",
          "SENS vs. the hallmarks of aging: competing visions, shared challenges",
          "CC BY"),
    Paper(3, "ajoolabady-2025-hallmarks-mechanisms-cellular-senescence", "PMC12322153",
          "10.1038/s41420-025-02655-x",
          "Hallmarks and mechanisms of cellular senescence in aging and disease",
          "CC BY"),
    Paper(4, "berardi-2026-shortest-telomere-drives-senescence", "PMC13168691",
          "10.1038/s41467-026-70352-z",
          "Both genome instability and replicative senescence stem from the shortest telomere",
          "CC BY"),
    Paper(5, "coquette-2026-telomere-dysfunction-proteostasis-senescence-pathways", "PMC13096579",
          "10.1111/acel.70512",
          "Telomere dysfunction and proteostasis decline define distinct pathways of senescence",
          "CC BY"),
    Paper(6, "sun-2026-nad-homeostasis-antiaging-framework", "PMC13144588",
          "10.1016/j.redox.2026.104191",
          "An integrated anti-aging framework targeting NAD+ homeostasis",
          "CC BY"),
    Paper(7, "levine-2018-phenoage-epigenetic-biomarker-lifespan", "PMC5940111",
          "10.18632/aging.101414",
          "An epigenetic biomarker of aging for lifespan and healthspan",
          "CC BY"),
    Paper(8, "belsky-2020-dunedin-pace-of-aging-blood-test", "PMC7282814",
          "10.7554/elife.54870",
          "Quantification of the pace of biological aging in humans through a blood test",
          "CC BY"),
    Paper(9, "vetter-2026-comparing-fourteen-biomarkers-of-aging", "PMC13063672",
          "10.1186/s40364-026-00909-z",
          "Comparing fourteen consensus biomarkers of aging",
          "CC BY"),
    Paper(10, "koch-2026-pan-epigenetic-age-prediction-mammals", "PMC12841597",
          "10.1111/acel.70380",
          "Pan-epigenetic age prediction in mammals",
          "CC BY"),
    Paper(11, "maleszka-2025-no-epigenetic-clock-in-insect", "PMC12557811",
          "10.1073/pnas.2523241122",
          "Still no evidence for an environmentally responsive epigenetic clock in an insect",
          "CC BY-NC-ND"),
    Paper(12, "mohammed-2021-metformin-anti-aging-critical-review", "PMC8374068",
          "10.3389/fendo.2021.718942",
          "A critical review of the evidence that metformin is a putative anti-aging drug",
          "CC BY"),
    Paper(13, "ivimey-cook-2025-rapamycin-not-metformin-dietary-restriction", "PMC12419861",
          "10.1111/acel.70131",
          "Rapamycin, not metformin, mirrors dietary restriction-driven lifespan extension",
          "CC BY"),
    Paper(14, "gkioni-2025-trametinib-rapamycin-combined-healthspan", "PMC12270913",
          "10.1038/s43587-025-00876-4",
          "Trametinib and rapamycin combine additively to extend mouse healthspan and lifespan",
          "CC BY"),
    Paper(15, "hickson-2019-senolytics-dasatinib-quercetin-first-in-human", "PMC6796530",
          "10.1016/j.ebiom.2019.08.069",
          "Senolytics decrease senescent cells in humans: dasatinib plus quercetin trial",
          "CC BY"),
    Paper(16, "justice-2019-senolytics-idiopathic-pulmonary-fibrosis-trial", "PMC6412088",
          "10.1016/j.ebiom.2018.12.052",
          "Senolytics in idiopathic pulmonary fibrosis: first-in-human, open-label pilot",
          "CC BY"),
    Paper(17, "kasamoto-2026-senolytics-do-not-reverse-senescence-methylation", "PMC12938503",
          "10.1111/acel.70430",
          "DNA methylation signatures of cellular senescence are not reversed by senolytics",
          "CC BY"),
    Paper(18, "moulds-2026-graded-calorie-restriction-epigenetic-ageing", "PMC12724011",
          "10.1111/acel.70342",
          "Graded calorie restriction causes graded slowing of epigenetic ageing in mice",
          "CC BY"),
    Paper(19, "schoenfeldt-2025-chemical-reprogramming-extends-lifespan", "PMC12340157",
          "10.1038/s44321-025-00265-9",
          "Chemical reprogramming ameliorates cellular hallmarks of aging and extends lifespan",
          "CC BY"),
    Paper(20, "mitchell-2026-chemical-reprogramming-lipid-droplet-toxicity", "PMC12835892",
          "10.1111/acel.70390",
          "In vivo chemical reprogramming causes a toxic accumulation of lipid droplets",
          "CC BY"),
    # Equation-focused batch — mathematical models of aging, added to stress-test
    # whether typeset math survives PDF parsing (see docstring above).
    Paper(21, "flietner-2025-unifying-theory-of-aging-mortality", "PMC12328599",
          "10.1038/s41598-025-11454-4",
          "A unifying theory of aging and mortality",
          "CC BY"),
    Paper(22, "nielsen-2024-gompertz-law-subcomponent-interdependencies", "PMC10786855",
          "10.1038/s41598-024-51669-5",
          "The Gompertz Law emerges naturally from the inter-dependencies between "
          "sub-components in complex organisms",
          "CC BY"),
    Paper(23, "pyrkov-2021-loss-of-resilience-lifespan-limit", "PMC8149842",
          "10.1038/s41467-021-23014-1",
          "Longitudinal analysis of blood markers reveals progressive loss of "
          "resilience and predicts human lifespan limit",
          "CC BY"),
    Paper(24, "oswal-2022-hierarchical-process-model-celegans-aging", "PMC9524676",
          "10.1371/journal.pcbi.1010415",
          "A hierarchical process model links behavioral aging and lifespan in C. elegans",
          "CC BY"),
    Paper(25, "farrell-2024-epigenetic-pacemaker-aging-moderators", "PMC10791860",
          "10.3389/fbinf.2023.1308680",
          "Identifying epigenetic aging moderators using the epigenetic pacemaker",
          "CC BY"),
    Paper(26, "dolejs-2025-strehler-mildvan-correlation-mortality", "PMC12528207",
          "10.3389/fpubh.2025.1627111",
          "The Strehler-Mildvan correlation as a valuable tool for monitoring the "
          "long-term health status of a population",
          "CC BY"),
    Paper(27, "gavrilova-2025-compensation-effect-of-mortality", "PMC12721371",
          "10.53941/alr.2025.100004",
          "The Compensation Effect of Mortality: A Global Analysis of Human Populations",
          "CC BY"),
)


class FetchError(Exception):
    """A paper could not be downloaded in a usable form."""


def download_pdf(pmcid: str) -> bytes:
    """Fetch one PDF, retrying past rate-limit responses.

    Raises FetchError when every attempt returns something that is not a PDF.
    """
    url = PDF_URL.format(pmcid=pmcid)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_problem = "no attempt made"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                body: bytes = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_problem = f"transport error: {exc}"
        else:
            if not body.startswith(b"%PDF-"):
                # Almost always the throttling response, which arrives as HTTP 200.
                preview = body[:60].decode("utf-8", errors="replace").strip()
                last_problem = f"not a PDF ({len(body)} bytes): {preview!r}"
            elif len(body) < MIN_PDF_BYTES:
                last_problem = f"suspiciously small ({len(body)} bytes)"
            else:
                return body

        if attempt < MAX_ATTEMPTS:
            wait = BACKOFF_SECONDS * attempt
            logger.warning("  %s attempt %d/%d failed (%s); retrying in %.0fs",
                           pmcid, attempt, MAX_ATTEMPTS, last_problem, wait)
            time.sleep(wait)

    raise FetchError(f"{pmcid}: {last_problem}")


def verify_pdf(path: Path) -> tuple[int, int]:
    """Open a PDF and extract text from every page.

    Returns (page count, extracted character count). Raises FetchError if the file
    cannot be parsed — a valid PDF header is not enough, since a malformed font
    descriptor can make extraction fail on an otherwise well-formed file.
    """
    try:
        reader = PdfReader(path)
        characters = sum(len(page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # pypdf raises a wide range of parse errors
        raise FetchError(f"{path.name}: unreadable ({type(exc).__name__}: {exc})") from exc

    if characters == 0:
        raise FetchError(f"{path.name}: no extractable text (scanned image?)")
    return len(reader.pages), characters


def fetch_one(paper: Paper, out_dir: Path, *, force: bool) -> tuple[int, int, bool]:
    """Ensure one paper is on disk and readable.

    Returns (pages, characters, downloaded). `downloaded` is False when a valid file
    was already present.
    """
    destination = out_dir / paper.filename

    if destination.exists() and not force:
        try:
            pages, characters = verify_pdf(destination)
        except FetchError as exc:
            logger.warning("  existing file is bad, re-downloading (%s)", exc)
        else:
            return pages, characters, False

    body = download_pdf(paper.pmcid)
    destination.write_bytes(body)
    try:
        pages, characters = verify_pdf(destination)
    except FetchError:
        destination.unlink(missing_ok=True)
        raise
    return pages, characters, True


def run(papers: tuple[Paper, ...], out_dir: Path, *, force: bool,
        verify_only: bool, delay: float) -> int:
    """Fetch or verify every paper. Returns a process exit code."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("corpus directory: %s", out_dir)
    logger.info("%d papers, %s\n", len(papers), "verify only" if verify_only else "fetch missing")

    total_pages = 0
    total_characters = 0
    total_bytes = 0
    downloaded = 0
    failures: list[str] = []

    for position, paper in enumerate(papers):
        label = f"[{paper.index:02d}/{len(PAPERS)}] {paper.title[:64]}"
        destination = out_dir / paper.filename

        if verify_only:
            if not destination.exists():
                logger.error("%s\n  MISSING", label)
                failures.append(f"{paper.filename}: missing")
                continue
            try:
                pages, characters = verify_pdf(destination)
            except FetchError as exc:
                logger.error("%s\n  BAD: %s", label, exc)
                failures.append(str(exc))
                continue
            was_downloaded = False
        else:
            try:
                pages, characters, was_downloaded = fetch_one(paper, out_dir, force=force)
            except FetchError as exc:
                logger.error("%s\n  FAILED: %s", label, exc)
                failures.append(str(exc))
                continue

        size = destination.stat().st_size
        total_pages += pages
        total_characters += characters
        total_bytes += size
        downloaded += int(was_downloaded)
        state = "downloaded" if was_downloaded else "already present"
        logger.info("%s\n  %s · %d pages · %s chars · %.1f MB · %s",
                    label, state, pages, f"{characters:,}", size / 1e6, paper.licence)

        # Space out requests; the source throttles bursts. No need to wait after the
        # last paper, or after one that was served from disk.
        if was_downloaded and position < len(papers) - 1:
            time.sleep(delay)

    logger.info("\n%s", "-" * 72)
    logger.info("%d/%d papers present · %d newly downloaded", len(papers) - len(failures),
                len(papers), downloaded)
    logger.info("%d pages · %s characters · %.1f MB", total_pages, f"{total_characters:,}",
                total_bytes / 1e6)
    if failures:
        logger.error("\n%d failed:", len(failures))
        for failure in failures:
            logger.error("  - %s", failure)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"corpus directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="re-download papers that are already present")
    parser.add_argument("--verify-only", action="store_true",
                        help="check files on disk without downloading anything")
    parser.add_argument("--only", type=int, nargs="+", metavar="N",
                        help="fetch only these corpus numbers (1-20)")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS,
                        help=f"seconds between downloads (default: {REQUEST_DELAY_SECONDS})")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    papers = PAPERS
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {paper.index for paper in PAPERS}
        if unknown:
            parser.error(f"no such corpus numbers: {sorted(unknown)}")
        papers = tuple(paper for paper in PAPERS if paper.index in wanted)

    return run(papers, args.out, force=args.force,
               verify_only=args.verify_only, delay=args.delay)


if __name__ == "__main__":
    raise SystemExit(main())
