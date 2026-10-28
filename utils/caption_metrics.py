"""Support the original bundled caption toolkit and Python 3 packages."""

try:
    from cococaption.pycocotools.coco import COCO
    from cococaption.pycocoevalcap.eval import COCOEvalCap
except ModuleNotFoundError:
    try:
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Caption evaluation requires cococaption in the repository root, '
            'or installed pycocotools and pycocoevalcap packages. '
            'METEOR/SPICE also require Java.'
        ) from exc
