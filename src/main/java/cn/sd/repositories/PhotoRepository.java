package cn.sd.repositories;

import cn.sd.entities.Photo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface PhotoRepository extends JpaRepository<Photo, Long> {
    List<Photo> findPhotoByTypeAndUserId(String type, Long userId);

    List<Photo> findPhotoByUserIdOrderByCreateTimeDesc(Long userId);

    @Query(value = "select * from photo where createTime like %?2% and userId = ?1 order by createTime desc", nativeQuery = true)
    List<Photo> findPhotoByUserIdAndCreateTime(Long userId, String time);

    @Query(value = "select *,max(position) from photo where userId=?1 and type in(select mainType from photoType)group by type", nativeQuery = true)
    List<Photo> findAllTypeOnePhotoByUserId(Long userId);

    @Query(value = "select * from photo where id in(select photoId from FaceClustering where userId = ?2 and clusteringId = ?1)", nativeQuery = true)
    List<Photo> findPhotoByClusteringIdAndUserId(Long clusteringId, Long userId);
}
