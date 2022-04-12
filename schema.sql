-- psql --host=opensea.ckmusmy93z05.eu-west-2.rds.amazonaws.com --port=5432 --username=marfapopova21 --password --dbname=opensea

drop schema nfts cascade;
create schema nfts;

drop table if exists nfts.assets cascade;
create table nfts.assets (
    id                      varchar(100),
    creator                 varchar(100),
    artwork_name            varchar(1000),
    collection              varchar(1000),
    price                   varchar(1000),
    currency                varchar(10),
    tokens                  numeric,
    likes                   varchar(4),
    nsfw                    boolean
);

drop table if exists nfts.collections cascade;
create table nfts.collections (
    collection_name         varchar(100),
    created_date            varchar(100),
    collection_status       varchar(100),
    nft_version             varchar(10),
    tokens                  varchar(100),
    owner_number            numeric,
    featured                boolean,
    hidden                  boolean,
    nsfw                    boolean
);

drop table if exists nfts.finances cascade;
create table nfts.finances (
    collection_name                     varchar(100),
    asset_contract_type                 varchar(100),
    require_email                       boolean,
    day_avg_price                       numeric,
    week_avg_price                      numeric,
    month_avg_price                     numeric,
    total_volume                        numeric,
    total_sales                         numeric,
    total_supply                        numeric,
    max_price                           numeric,
    min_price                           numeric,
    average_price                       numeric,
    only_proxied_transfers              boolean,
    is_subject_to_whitelist             boolean,
    opensea_buyer_fee_basis_points      varchar(5),
    opensea_seller_fee_basis_points     varchar(5)
);

drop table if exists nfts.urls cascade;
create table nfts.urls (
    image_url               varchar(1000),
    large_image_url         varchar(1000),
    slug                    varchar(100),
    wiki                    varchar(1000)
);

drop table if exists nfts.socials cascade;
create table nfts.socials (
    telegram               varchar(1000),
    twitter                varchar(1000),
    instagram              varchar(1000)
);

\dn
\dt nfts.*
select * from nfts.assets;
select count(*) from nfts.assets;