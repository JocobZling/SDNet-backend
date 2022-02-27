package cn.sd.repositories;

import cn.sd.entities.Clustering;
import feign.Param;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;

import javax.transaction.Transactional;
import java.util.List;

public interface ClusteringRepository extends JpaRepository<Clustering, Long> {

    @Query(value = "SELECT id FROM Clustering")
    List<Long> findAllIds();

    @Query(value = "SELECT clusterName FROM Clustering WHERE id =:id AND userId =:userId")
    String findName(@Param("id") Long id, @Param("userId") Long userId);

    List<String> findClusteringByUserIdAndId(Long userId,Long Id);
    @Transactional
    @Modifying
    @Query(value = "update Clustering  SET clusterName =:clusterName where userId =:userId and id = :id")
    void updateClusterName(@Param("id") Long id, @Param("userId") Long userId, @Param("clusterName") String clusterName);

}
