import {
    CredentialsProvider,
    makeCredentialsGetter,
} from "#src/credentials_provider/index.js";
import { getCredentialsWithStatus } from "#src/credentials_provider/interactive_credentials_provider.js";
import type { CatmaidToken } from "#src/datasource/catmaid/api.js";
import { fetchOk } from "#src/util/http_request.js";
import { ProgressSpan } from "#src/util/progress_listener.js";

async function getAnonymousToken(
    serverUrl: string,
    signal: AbortSignal,
): Promise<CatmaidToken> {
    // serverUrl passed here is the base URL.

    const tokenUrl = `${serverUrl}/accounts/anonymous-api-token`;

    const response = await fetchOk(tokenUrl, {
        method: "GET",
        signal: signal,
    });

    const json = await response.json();
    if (typeof json === 'object' && json !== null && typeof json.token === 'string') {
        return { token: json.token };
    }
    throw new Error(`Unexpected response from ${tokenUrl}: ${JSON.stringify(json)}`);
}

export class CatmaidCredentialsProvider extends CredentialsProvider<CatmaidToken> {
    constructor(public serverUrl: string) {
        super();
    }

    get = makeCredentialsGetter(async (options) => {
        using _span = new ProgressSpan(options.progressListener, {
            message: `Requesting CATMAID access token from ${this.serverUrl}`,
        });
        return await getCredentialsWithStatus(
            {
                description: `CATMAID server ${this.serverUrl}`,
                supportsImmediate: true,
                get: async (signal, immediate) => {
                    if (immediate) {
                        try {
                            return await getAnonymousToken(this.serverUrl, signal);
                        } catch (e) {
                            throw e;
                        }
                    }
                    return await getAnonymousToken(this.serverUrl, signal);
                },
            },
            options.signal,
        );
    });
}
