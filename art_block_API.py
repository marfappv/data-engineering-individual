#https://docs.artblocks.io/creator-docs/art-blocks-api/api-overview/

{
  projects(where: {projectId: "0", contract_in: ["0x059edd72cd353df5106d2b9cc5ab83a52287ac3a", "0xa7d8d9ef8d8ce8992df33d8b8cf4aebabd5bd270"]}) {
    id
    invocations
    artistName
    name
  }
}

