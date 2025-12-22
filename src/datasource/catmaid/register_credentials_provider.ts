import { registerDefaultCredentialsProvider } from "#src/credentials_provider/default_manager.js";
import { credentialsKey } from "#src/datasource/catmaid/api.js";
import { CatmaidCredentialsProvider } from "#src/datasource/catmaid/credentials_provider.js";

registerDefaultCredentialsProvider(
    credentialsKey,
    (params: { serverUrl: string }) =>
        new CatmaidCredentialsProvider(params.serverUrl),
);
