CREATE TABLE `faceClustering` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `photoId` int(11),
    `position` varchar(200),
    `clusteringId` int(11),
    `userId` int(11),
    `airFaceId` varchar(200),
    `createTime` TIMESTAMP default current_timestamp,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
