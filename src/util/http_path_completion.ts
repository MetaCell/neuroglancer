/**
 * @license
 * Copyright 2019 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import type { CredentialsManager } from "#src/credentials_provider/index.js";
import type {
  BasicCompletionResult,
  Completion,
  CompletionWithDescription,
} from "#src/util/completion.js";
import { getPrefixMatchesWithDescriptions } from "#src/util/completion.js";
import { getGcsPathCompletions } from "#src/util/gcs_bucket_listing.js";
import { parseUrl } from "#src/util/http_request.js";
import { getS3PathCompletions } from "#src/util/s3.js";
import { getS3CompatiblePathCompletions } from "#src/util/s3_bucket_listing.js";
import type { SpecialProtocolCredentialsProvider } from "#src/util/special_protocol_request.js";
import {
  fetchSpecialOk,
  parseSpecialUrl,
} from "#src/util/special_protocol_request.js";

/**
 * Obtains a directory listing from a server that supports HTML directory listings.
 */
export async function getHtmlDirectoryListing(
  url: string,
  abortSignal: AbortSignal,
  credentialsProvider?: SpecialProtocolCredentialsProvider,
): Promise<string[]> {
  const response = await fetchSpecialOk(
    credentialsProvider,
    url,
    /*init=*/ { headers: { accept: "text/html" }, signal: abortSignal },
  );
  const contentType = response.headers.get("content-type");
  if (contentType === null || /\btext\/html\b/i.exec(contentType) === null) {
    return [];
  }
  const text = await response.text();
  const doc = new DOMParser().parseFromString(text, "text/html");
  const nodes = doc.evaluate(
    "//a/@href",
    doc,
    null,
    XPathResult.UNORDERED_NODE_SNAPSHOT_TYPE,
    null,
  );
  const results: string[] = [];
  for (let i = 0, n = nodes.snapshotLength; i < n; ++i) {
    const node = nodes.snapshotItem(i)!;
    const href = node.textContent;
    if (href) {
      results.push(new URL(href, url).toString());
    }
  }
  return results;
}

export async function getHtmlPathCompletions(
  url: string,
  abortSignal: AbortSignal,
  credentialsProvider?: SpecialProtocolCredentialsProvider,
): Promise<BasicCompletionResult> {
  console.log("getHtmlPathCompletions");
  const m = url.match(/^([a-z]+:\/\/.*\/)([^/?#]*)$/);
  if (m === null) throw null;
  const entries = await getHtmlDirectoryListing(
    m[1],
    abortSignal,
    credentialsProvider,
  );
  const offset = m[1].length;
  const matches: Completion[] = [];
  for (const entry of entries) {
    if (!entry.startsWith(url)) continue;
    matches.push({ value: entry.substring(offset) });
  }
  return {
    offset,
    completions: matches,
  };
}

const specialProtocolEmptyCompletions: CompletionWithDescription[] = [
  { value: "gs://", description: "Google Cloud Storage (JSON API)" },
  { value: "gs+xml://", description: "Google Cloud Storage (XML API)" },
  {
    value: "gs+ngauth+http://",
    description: "Google Cloud Storage (JSON API) authenticated via ngauth",
  },
  {
    value: "gs+ngauth+https://",
    description: "Google Cloud Storage (JSON API) authenticated via ngauth",
  },
  {
    value: "gs+xml+ngauth+http://",
    description: "Google Cloud Storage (XML API) authenticated via ngauth",
  },
  {
    value: "gs+xml+ngauth+https://",
    description: "Google Cloud Storage (XML API) authenticated via ngauth",
  },
  { value: "s3://", description: "Amazon Simple Storage Service (S3)" },
  { value: "https://" },
  { value: "http://" },
];

export async function completeHttpPath(
  credentialsManager: CredentialsManager,
  url: string,
  abortSignal: AbortSignal,
): Promise<BasicCompletionResult<Completion>> {
  if (!url.includes("://")) {
    return {
      offset: 0,
      completions: getPrefixMatchesWithDescriptions(
        url,
        specialProtocolEmptyCompletions,
        (x) => x.value,
        (x) => x.description,
      ),
    };
  }
  const { url: parsedUrl, credentialsProvider } = parseSpecialUrl(
    url,
    credentialsManager,
  );
  const offset = url.length - parsedUrl.length;
  let result;
  try {
    result = parseUrl(parsedUrl);
  } catch {
    throw null;
  }
  const { protocol, host, path } = result;
  const completions = await (async () => {
    if (protocol === "gs+xml" && path.length > 0) {
      return await getS3CompatiblePathCompletions(
        credentialsProvider,
        `${protocol}://${host}`,
        `https://storage.googleapis.com/${host}`,
        path,
        abortSignal,
      );
    }
    if (protocol === "gs" && path.length > 0) {
      return await getGcsPathCompletions(
        credentialsProvider,
        `${protocol}://${host}`,
        host,
        path,
        abortSignal,
      );
    }
    if (protocol === "s3" && path.length > 0) {
      return await getS3PathCompletions(host, path, abortSignal);
    }
    const s3Match = parsedUrl.match(
      /^((?:http|https):\/\/(?:storage\.googleapis\.com\/[^/]+|[^/]+\.storage\.googleapis\.com|[^/]+\.s3(?:[^./]+)?\.amazonaws.com))(\/.*)$/,
    );
    if (s3Match !== null) {
      return await getS3CompatiblePathCompletions(
        credentialsProvider,
        s3Match[1],
        s3Match[1],
        s3Match[2],
        abortSignal,
      );
    }
    if ((protocol === "http" || protocol === "https") && path.length > 0) {
      return await getHtmlPathCompletions(
        parsedUrl,
        abortSignal,
        credentialsProvider,
      );
    }
    throw null;
  })();
  return {
    offset: offset + completions.offset,
    completions: completions.completions,
  };
}
