from fastapi import APIRouter, Depends
from app.api.deps import get_clustering_service
from app.schemas import ClusterInitResponse
from app.services.clustering import ClusteringService

router = APIRouter()


@router.post("/cluster/init", response_model=ClusterInitResponse)
def init_cluster(
    season_id: int,
    k: int = 5,
    service: ClusteringService = Depends(get_clustering_service),
):
    return service.init_clusters(season_id, k)
