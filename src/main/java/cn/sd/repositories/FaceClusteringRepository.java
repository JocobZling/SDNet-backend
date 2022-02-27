package cn.sd.repositories;

import cn.sd.entities.FaceClustering;
import feign.Param;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface FaceClusteringRepository extends JpaRepository<FaceClustering, Long> {

    FaceClustering findByPhotoId(Long photoId);

    @Query(value = "select *,max(position) from FaceClustering where clusteringId in(select id from Clustering where userId = ?1)group by clusteringId", nativeQuery = true)
    List<FaceClustering> findOneFaceClustering(Long userId);

    @Query(value = "select position from FaceClustering where clusteringId=:clusteringId and userId=:userId")
    List<String> findPositionWhereUserIdAndClusteringId(@Param("clusteringId") Long clusteringId, @Param("userId") Long userId);

    FaceClustering findTopByAirFaceId(String airFaceId);

}
